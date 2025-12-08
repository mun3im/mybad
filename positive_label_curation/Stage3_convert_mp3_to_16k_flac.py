#!/usr/bin/env python3
"""
convert_to_16k_flac.py

Traverse an input root (species subfolders), convert each audio file to:
  - mono
  - 16000 Hz sample rate
  - 16-bit FLAC

Features:
  - Uses ffmpeg for robust decoding/encoding.
  - Uses soundfile + numpy to inspect the decoded audio for peak amplitude (clipping).
  - If clipping detected (peak >= 1.0), computes and applies a gain < 1 to avoid clipping.
  - Optional peak normalization to a target dBFS (e.g., -1.0 dBFS).
  - Parallel processing using ThreadPoolExecutor.
  - CSV log written with per-file info: detected_peak, gain_applied, clipped_flag, status, error.
  - Safe behavior: by default DOES NOT boost audio above 0 dBFS (only attenuation unless --allow-boost).

Usage:
    python convert_to_16k_flac.py \
      --inroot /Volumes/Evo/XC-All-Malaysian-Birds \
      --outroot /Volumes/Evo/XC-All-Malaysian-Birds-16k-flac \
      --workers 6 \
      --normalize_db -1.0

Notes:
 - ffmpeg must be on PATH.
 - temp intermediate WAV files are written and removed.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import csv
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import pandas as pd

# ---------------- DEFAULTS ----------------
DEFAULT_INROOT = "/Volumes/Evo/XC-All-Malaysian-Birds"
DEFAULT_OUTROOT = "/Volumes/Evo/XC-All-Malaysian-Birds-16k-flac"
TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "flac"
TARGET_SUBTYPE = "PCM_16"   # 16-bit
LOG_CSV = "conversion_log.csv"
# ------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("conv16k")

# ---------- Helper audio functions ----------

def run_ffmpeg_decode_to_wav(in_path: str, out_wav: str, to_mono: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Use ffmpeg to decode input file to a WAV at original sample rate and channels.
    We decode to PCM WAV for stable reading by soundfile.
    Returns (ok, error_message).
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vn",  # no video
    ]
    # write PCM signed 16-bit WAV
    cmd += ["-f", "wav"]
    if to_mono:
        cmd += ["-ac", "1"]
    cmd += [out_wav]
    try:
        subprocess.run(cmd, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg decode error: {e}"
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"


def run_ffmpeg_encode_flac(in_wav: str, out_path: str, sr: int = TARGET_SR, channels: int = TARGET_CHANNELS) -> Tuple[bool, Optional[str]]:
    """
    Use ffmpeg to encode WAV file to FLAC with desired samplerate and channels.
    Returns (ok, error_message).
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_wav,
        "-ac", str(channels),
        "-ar", str(sr),
        "-compression_level", "5",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg encode error: {e}"
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"


def measure_peak_from_wav(wav_path: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Read WAV using soundfile and return peak absolute amplitude (float).
    Returns (peak, error_message). peak is in linear PCM units (1.0 == full scale).
    """
    try:
        data, sr = sf.read(wav_path, always_2d=False, dtype="float32")
        if data is None:
            return None, "soundfile read returned None"
        if data.ndim > 1:
            # if multiple channels, compute max abs across all channels
            peak = float(np.max(np.abs(data)))
        else:
            peak = float(np.max(np.abs(data)))
        return peak, None
    except Exception as e:
        return None, f"soundfile read error: {e}"


def apply_gain_with_ffmpeg(orig_path: str, out_wav: str, gain: float) -> Tuple[bool, Optional[str]]:
    """
    Use ffmpeg to apply linear gain (volume filter) to orig_path and write out_wav (wav PCM).
    gain is linear factor (e.g., 0.8).
    """
    # ffmpeg volume filter accepts 'volume=G' where G is factor or dB.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", orig_path,
        "-af", f"volume={gain}",
        "-f", "wav",
        out_wav
    ]
    try:
        subprocess.run(cmd, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg volume error: {e}"
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"


# ---------- Core file processing ----------

def process_file(
    in_path: str,
    out_root: str,
    rel_subdir: str,
    normalize_db: Optional[float] = None,
    allow_boost: bool = False,
    tmp_dir: Optional[str] = None
) -> Dict:
    """
    Convert a single input audio file:
      1. decode to temp WAV
      2. measure peak
      3. if clipped (peak >= 1.0) compute attenuation factor and apply
      4. optional normalization (peak normalization to normalize_db dBFS)
      5. encode to FLAC 16k mono PCM_16
    Returns a log dict with keys:
      input_path, output_path, detected_peak, gain_applied, clipped_flag, status, error
    """
    log = {
        "input_path": in_path,
        "output_path": None,
        "detected_peak": None,
        "gain_applied": None,
        "clipped_flag": False,
        "status": "error",
        "error": None
    }
    try:
        # derive output path
        basename = os.path.basename(in_path)
        name_wo_ext = os.path.splitext(basename)[0]
        out_dir = os.path.join(out_root, rel_subdir)
        os.makedirs(out_dir, exist_ok=True)
        # target filename: same base but extension .flac
        out_path = os.path.join(out_dir, f"{name_wo_ext}.flac")
        log["output_path"] = out_path

        # If already exists, skip
        if os.path.exists(out_path):
            log["status"] = "skipped_exists"
            return log

        # create temp WAV decoded from original (mono not enforced yet)
        tmp_wav = os.path.join(tmp_dir or tempfile.gettempdir(), f"dec_{int(time.time()*1000)}_{os.getpid()}.wav")
        ok, err = run_ffmpeg_decode_to_wav(in_path, tmp_wav, to_mono=False)
        if not ok:
            log["error"] = f"decode_failed: {err}"
            return log

        # measure peak
        peak, err = measure_peak_from_wav(tmp_wav)
        if err:
            log["error"] = f"measure_peak_failed: {err}"
            # cleanup tmp wav
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
            return log
        log["detected_peak"] = float(peak)

        work_wav = tmp_wav  # current wav to use for final encoding (may be replaced if we apply gain)

        # clipping detection
        if peak >= 1.0:
            log["clipped_flag"] = True
            # compute attenuation factor so max becomes 0.99 (to avoid clip)
            target_peak = 0.99
            att = target_peak / peak
            # apply attenuation via ffmpeg to a new temp wav
            tmp_wav2 = os.path.join(tmp_dir or tempfile.gettempdir(), f"dec_gain_{int(time.time()*1000)}_{os.getpid()}.wav")
            ok2, err2 = apply_gain_with_ffmpeg(tmp_wav, tmp_wav2, att)
            if not ok2:
                log["error"] = f"apply_gain_failed: {err2}"
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
                return log
            # measure peak again to record what we achieved
            new_peak, err3 = measure_peak_from_wav(tmp_wav2)
            if err3:
                log["error"] = f"measure_peak_after_gain_failed: {err3}"
                try:
                    os.remove(tmp_wav); os.remove(tmp_wav2)
                except Exception:
                    pass
                return log
            log["gain_applied"] = float(att)
            log["detected_peak"] = float(new_peak)
            work_wav = tmp_wav2
            # remove original tmp_wav
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

        # optional normalization (peak normalization to normalize_db)
        if normalize_db is not None:
            # convert dBFS to linear target_peak (1.0 is 0 dBFS)
            target_linear = 10 ** (normalize_db / 20.0)
            measured_peak = log["detected_peak"] if log["detected_peak"] is not None else (measure_peak_from_wav(work_wav)[0] or 0.0)
            if measured_peak <= 0:
                # nothing to normalize
                pass
            else:
                gain_needed = target_linear / measured_peak
                # if boosting is not allowed and gain_needed > 1, cap to 1.0
                if (not allow_boost) and gain_needed > 1.0:
                    gain_needed = 1.0
                if gain_needed != 1.0:
                    tmp_wav3 = os.path.join(tmp_dir or tempfile.gettempdir(), f"norm_{int(time.time()*1000)}_{os.getpid()}.wav")
                    okn, errn = apply_gain_with_ffmpeg(work_wav, tmp_wav3, gain_needed)
                    if not okn:
                        log["error"] = f"normalize_gain_failed: {errn}"
                        try:
                            os.remove(work_wav)
                        except Exception:
                            pass
                        return log
                    new_peak2, errn2 = measure_peak_from_wav(tmp_wav3)
                    if errn2:
                        log["error"] = f"measure_peak_after_normalize_failed: {errn2}"
                        try:
                            os.remove(work_wav); os.remove(tmp_wav3)
                        except Exception:
                            pass
                        return log
                    # record cumulative gain (multiply if there was earlier gain)
                    if log["gain_applied"] is None:
                        log["gain_applied"] = float(gain_needed)
                    else:
                        log["gain_applied"] = float(log["gain_applied"] * gain_needed)
                    log["detected_peak"] = float(new_peak2)
                    # remove previous work_wav if it was a temp file
                    try:
                        if work_wav != tmp_wav2 and os.path.exists(work_wav) and work_wav.startswith(tempfile.gettempdir()):
                            os.remove(work_wav)
                    except Exception:
                        pass
                    work_wav = tmp_wav3

        # final encode to flac 16k mono
        # if work_wav might still be stereo -> let ffmpeg convert to mono in encoding step
        okf, errf = run_ffmpeg_encode_flac(work_wav, out_path, sr=TARGET_SR, channels=TARGET_CHANNELS)
        if not okf:
            log["error"] = f"final_encode_failed: {errf}"
            try:
                os.remove(work_wav)
            except Exception:
                pass
            return log

        # cleanup temp wav(s)
        try:
            if os.path.exists(work_wav):
                os.remove(work_wav)
        except Exception:
            pass

        log["status"] = "ok"
        return log

    except Exception as e:
        log["error"] = f"unexpected_exception: {e}"
        return log


# ---------- Directory traversal and parallelization ----------

def discover_input_files(inroot: str) -> list:
    """
    Walk inroot and return list of (full_path, relative_subdir) pairs.
    relative_subdir is the path under inroot to use when creating output subdir.
    """
    pairs = []
    for root, dirs, files in os.walk(inroot):
        # compute relative path under inroot
        rel = os.path.relpath(root, inroot)
        # normalize reldir to '' for root
        if rel == ".":
            rel = ""
        for fn in files:
            # skip hidden files
            if fn.startswith("."):
                continue
            full = os.path.join(root, fn)
            pairs.append((full, rel))
    return pairs


def worker_wrapper(args_tuple):
    return process_file(*args_tuple)


def convert_all(
    inroot: str,
    outroot: str,
    workers: int = 4,
    normalize_db: Optional[float] = None,
    allow_boost: bool = False,
    log_csv_path: str = LOG_CSV
):
    items = discover_input_files(inroot)
    logger.info(f"Discovered {len(items)} files under {inroot}")
    os.makedirs(outroot, exist_ok=True)
    # prepare tmp dir for worker writes
    tmp_dir = tempfile.mkdtemp(prefix="convtmp_")
    logger.info(f"Using tmp dir: {tmp_dir}")

    # prepare tasks
    tasks = []
    for full, rel in items:
        tasks.append((full, outroot, rel, normalize_db, allow_boost, tmp_dir))

    # results list to write to CSV
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_wrapper, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.get("status") == "ok":
                logger.info(f"OK: {res.get('input_path')} -> {res.get('output_path')} peak={res.get('detected_peak'):.4f} gain={res.get('gain_applied')}")
            else:
                logger.warning(f"ERR: {res.get('input_path')} status={res.get('status')} error={res.get('error')}")

    # write CSV log
    df = pd.DataFrame(results)
    df.to_csv(log_csv_path, index=False)
    logger.info(f"Wrote conversion log to {log_csv_path}")

    # cleanup tmp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Convert audio collection to mono 16k FLAC with clipping detection, optional normalization, and parallel processing.")
    p.add_argument("--inroot", default=DEFAULT_INROOT, help="Input root folder (species subfolders)")
    p.add_argument("--outroot", default=DEFAULT_OUTROOT, help="Output root folder (mirrors subfolders)")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads")
    p.add_argument("--normalize-db", type=float, default=None, help="If set, peak-normalize to this dBFS (e.g. -1.0). Use None to disable.")
    p.add_argument("--allow-boost", action="store_true", help="Allow boosting (gain>1) during normalization. Default: disabled (only attenuation).")
    p.add_argument("--log-csv", default=LOG_CSV, help="CSV path for conversion log.")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Starting conversion")
    logger.info(f"Input root: {args.inroot}")
    logger.info(f"Output root: {args.outroot}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Normalize dB: {args.normalize_db} allow_boost: {args.allow_boost}")
    convert_all(
        inroot=args.inroot,
        outroot=args.outroot,
        workers=args.workers,
        normalize_db=args.normalize_db,
        allow_boost=args.allow_boost,
        log_csv_path=args.log_csv
    )
    logger.info("All done.")


if __name__ == "__main__":
    main()
