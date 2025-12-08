#!/usr/bin/env python3
"""
xc_download_my_birds_fixed.py

Improved downloader for Xeno-Canto metadata CSV -> media files.

Fixes and features:
 - robust handling when CSV cells are NaN/float (coerce to string safely)
 - appends quality: filename format `xc{ID}_{Q}{ext}` (e.g. xc422286_U.mp3)
 - writes an append-only download log CSV (download_log.csv) with status and errors
 - validates downloaded files by minimal size threshold and removes tiny files
 - uses .part temp files and resumes safely (skips existing final files)
"""

import argparse
import os
import time
import logging
import sys
from typing import Optional
import requests
import pandas as pd
from urllib.parse import urlparse, unquote
import csv

# --------- Config defaults ----------
DEFAULT_OUT_ROOT = "/Volumes/Evo/XC-All-Malaysian-Birds"
RATE_LIMIT_DELAY = 0.1   # seconds between requests (politeness)
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30     # seconds per request
CHUNK_SIZE = 1024 * 64   # 64KB
USER_AGENT = "xc_downloader/1.0 (+https://your.email@example.com)"
DOWNLOAD_LOG = os.path.join(DEFAULT_OUT_ROOT, "download_log.csv")
MIN_BYTES_ACCEPTED = 1024  # 1 KB minimal acceptable file size (tweak as needed)
# ------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xc_downloader_fixed")


def sanitize_folder_name(name: str) -> str:
    """Make a filesystem-safe folder name from the species English name."""
    if not isinstance(name, str) or name.strip() == "":
        return "unknown_species"
    s = name.strip()
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    return s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_extension_from_url(url: str, fallback: str = ".mp3") -> str:
    """Try to extract sensible extension from the URL path. Fallback to .mp3."""
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        basename = os.path.basename(path)
        if "." in basename:
            ext = os.path.splitext(basename)[1]
            if ext:
                return ext
    except Exception:
        pass
    return fallback


def download_url_to_path(url: str, out_path: str, max_retries: int = MAX_RETRIES, timeout: int = REQUEST_TIMEOUT) -> (bool, str, int, float):
    """
    Download URL streaming to out_path with retries.
    Returns tuple: (success, error_message_or_empty, bytes_written, elapsed_seconds)
    - Writes to out_path + ".part" then renames on success.
    """
    headers = {"User-Agent": USER_AGENT}
    attempt = 0
    start_time_total = time.time()
    tmp_path = out_path + ".part"

    while attempt <= max_retries:
        try:
            if attempt > 0:
                backoff = 2 ** (attempt - 1)
                logger.info(f"Retrying after {backoff}s (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)

            time.sleep(RATE_LIMIT_DELAY)

            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    msg = f"Non-200 status {r.status_code}"
                    logger.warning(f"{msg} for URL {url} (resp text start: {r.text[:200]!r})")
                    # 4xx likely won't succeed by retrying
                    if 400 <= r.status_code < 500:
                        return False, f"{msg} (client error)", 0, time.time() - start_time_total
                    attempt += 1
                    continue

                # stream to temp file
                with open(tmp_path, "wb") as fout:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fout.write(chunk)
                        downloaded += len(chunk)

                # basic size check
                try:
                    actual = os.path.getsize(tmp_path)
                except Exception:
                    actual = downloaded

                if actual < MIN_BYTES_ACCEPTED:
                    # tiny file -> treat as failure (sometimes API returns small HTML error)
                    logger.warning(f"Downloaded tiny file {actual} bytes for URL {url} -> rejecting")
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    attempt += 1
                    continue

                # rename to final
                os.replace(tmp_path, out_path)
                elapsed = time.time() - start_time_total
                return True, "", actual, elapsed

        except requests.exceptions.RequestException as e:
            logger.warning(f"RequestException for URL {url}: {e}")
            attempt += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error while downloading {url}: {e}")
            attempt += 1
            continue

    elapsed = time.time() - start_time_total
    return False, f"Failed after {max_retries} retries", 0, elapsed


def safe_str(val) -> str:
    """
    Safely convert a CSV cell value to stripped string.
    Handles NaN / float gracefully.
    """
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    try:
        return str(val).strip()
    except Exception:
        return ""


def append_log_row(log_path: str, row: dict):
    """Append one row (dict) to download_log.csv; create header if missing."""
    ensure_dir(os.path.dirname(log_path))
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(lf, fieldnames=[
            "id", "en", "file_url", "q", "out_path", "status", "error", "bytes", "elapsed_s", "ts"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_csv_and_download(csv_path: str, out_root: str, dry_run: bool = False, limit: Optional[int] = None):
    """Read CSV and download files according to rules (robust to NaN)."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(2)

    # read CSV (object dtype so NaN preserved; we'll coerce safely)
    df = pd.read_csv(csv_path, dtype=object)

    total = len(df)
    logger.info(f"Loaded {total} rows from {csv_path}")

    count = 0
    downloaded = 0
    skipped_no_url = 0
    skipped_exists = 0
    failed = 0

    for idx, row in df.iterrows():
        if limit is not None and count >= limit:
            break
        count += 1

        # safe extraction
        rec_id = safe_str(row.get("id", ""))
        en = safe_str(row.get("en", ""))
        file_url = safe_str(row.get("file", ""))
        q = safe_str(row.get("q", ""))

        if rec_id == "":
            logger.warning(f"Row {idx}: missing id - skipping")
            failed += 1
            append_log_row(DOWNLOAD_LOG, {
                "id": "", "en": en, "file_url": file_url, "q": q, "out_path": "",
                "status": "skip", "error": "missing id", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        # skip if file URL missing
        if not file_url or file_url.upper() in ("NULL", "NONE", "NAN", "NA"):
            skipped_no_url += 1
            logger.info(f"Row id={rec_id}: no file URL (skipping)")
            append_log_row(DOWNLOAD_LOG, {
                "id": rec_id, "en": en, "file_url": file_url, "q": q, "out_path": "",
                "status": "skip", "error": "no_url", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        # quality char (append)
        q_char = q[0] if q else "U"

        # id canonical digits if possible
        try:
            id_int = int(float(rec_id))
            id_str = str(int(id_int))
        except Exception:
            digits = "".join(ch for ch in rec_id if ch.isdigit())
            id_str = digits if digits else rec_id

        species_folder = sanitize_folder_name(en) if en else "unknown_species"
        dest_folder = os.path.join(out_root, species_folder)
        ensure_dir(dest_folder)

        ext = get_extension_from_url(file_url, fallback=".mp3")
        out_fname = f"xc{id_str}_{q_char}{ext}"
        out_path = os.path.join(dest_folder, out_fname)

        if os.path.exists(out_path):
            skipped_exists += 1
            logger.info(f"Already exists: {out_path} (skipping)")
            append_log_row(DOWNLOAD_LOG, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path,
                "status": "exists", "error": "", "bytes": os.path.getsize(out_path), "elapsed_s": 0.0, "ts": time.time()
            })
            continue

        if dry_run:
            logger.info(f"[DRY] Would download: {file_url} -> {out_path}")
            append_log_row(DOWNLOAD_LOG, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path,
                "status": "dry", "error": "", "bytes": 0, "elapsed_s": 0.0, "ts": time.time()
            })
            downloaded += 1
            continue

        logger.info(f"Downloading ({count}/{total}): id={id_str} q={q_char} -> {out_path}")
        ok, err_msg, bytes_written, elapsed = download_url_to_path(file_url, out_path)
        if ok:
            downloaded += 1
            logger.info(f"Saved: {out_path} ({bytes_written} bytes)")
            append_log_row(DOWNLOAD_LOG, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path,
                "status": "ok", "error": "", "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
        else:
            failed += 1
            logger.error(f"Failed to download id={id_str} from {file_url}: {err_msg}")
            append_log_row(DOWNLOAD_LOG, {
                "id": id_str, "en": en, "file_url": file_url, "q": q_char, "out_path": out_path,
                "status": "fail", "error": err_msg, "bytes": bytes_written, "elapsed_s": round(elapsed, 2), "ts": time.time()
            })
            # ensure no tiny leftover
            try:
                tmp = out_path + ".part"
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    logger.info("=== Summary ===")
    logger.info(f"Rows scanned: {count}")
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Skipped (no URL): {skipped_no_url}")
    logger.info(f"Skipped (already exists): {skipped_exists}")
    logger.info(f"Failed downloads: {failed}")
    logger.info(f"Download log (if created) at: {DOWNLOAD_LOG}")


def parse_cmdline():
    p = argparse.ArgumentParser(description="Download Xeno-Canto media files organized by species folder.")
    p.add_argument("--csv", required=True, help="Path to input metadata CSV (must contain id,en,file,q columns)")
    p.add_argument("--out", required=False, default=DEFAULT_OUT_ROOT, help=f"Output root folder (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--dry-run", action="store_true", help="Don't download files; just show what would be done")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many rows (useful for testing/resume)")
    return p.parse_args()


def main():
    args = parse_cmdline()
    out_root = args.out
    csv_path = args.csv
    dry_run = args.dry_run
    limit = args.limit

    global DOWNLOAD_LOG
    DOWNLOAD_LOG = os.path.join(out_root, "download_log.csv")

    logger.info(f"Output root: {out_root}")
    ensure_dir(out_root)
    process_csv_and_download(csv_path=csv_path, out_root=out_root, dry_run=dry_run, limit=limit)


if __name__ == "__main__":
    main()
