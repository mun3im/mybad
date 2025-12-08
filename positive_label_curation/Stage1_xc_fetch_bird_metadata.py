#!/usr/bin/env python3
"""
xc_fetch_country_birds.py

Fetch ALL Xeno-Canto v3 recordings for a given country (Malaysia or Singapore),
filter to records whose metadata indicates they are birds, and save full metadata to CSV.

Requirements:
  - requests
  - pandas

Set XENO_API_KEY environment variable before running:
  export XENO_API_KEY="your_api_key_here"

# Fetch Malaysia birds
python xc_fetch_country_birds.py --country Malaysia --out-csv xc_my_birds.csv

# Fetch Singapore birds
python xc_fetch_country_birds.py --country Singapore --out-csv xc_sg_birds.csv

"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional
import requests
import pandas as pd
import argparse

# ------------ CONFIG -------------
OUT_CSV_DEFAULT = "xc_country_birds.csv"
BASE_URL = "https://xeno-canto.org/api/3/recordings"
RATE_LIMIT_DELAY = 0.2
MAX_RETRIES = 4
REQUEST_TIMEOUT = 30
GROUP_FIELD_CANDIDATES = ["grp", "group", "animal", "type", "kind"]
# ----------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xc_country_birds")


def request_with_retries(url: str, params: Dict[str, str], headers: Dict[str, str], retries: int = 0) -> Optional[requests.Response]:
    """GET with retries and exponential backoff for transient errors."""
    try:
        time.sleep(RATE_LIMIT_DELAY)
        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r
        if 400 <= r.status_code < 500:
            logger.warning(f"Client error {r.status_code} for URL {r.url} - response start: {r.text[:200]!r}")
            return None
        if 500 <= r.status_code < 600:
            raise requests.exceptions.HTTPError(f"Server error {r.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            backoff = 2 ** retries
            logger.warning(f"Request failed (attempt {retries+1}) -> retrying in {backoff}s: {e}")
            time.sleep(backoff)
            return request_with_retries(url, params, headers, retries + 1)
        else:
            logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
            return None


def looks_like_bird(record: Dict) -> bool:
    """Heuristic to check candidate fields for bird indicators."""
    for f in GROUP_FIELD_CANDIDATES:
        if f in record and record[f] is not None:
            try:
                v = str(record[f]).strip().lower()
            except Exception:
                continue
            if v == "":
                continue
            if "bird" in v or "aves" in v or "avian" in v:
                return True
    return False


def fetch_country_birds(api_key: str, country: str) -> List[Dict]:
    """Fetch all recordings for a given country and filter bird records."""
    headers = {"User-Agent": "xc_country_birds_fetcher/1.0", "Accept": "application/json"}
    query_tag = f'cnt:"{country}"'
    page = 1
    all_bird_records: List[Dict] = []
    total_fetched = 0

    while True:
        params = {"key": api_key, "query": query_tag, "page": page}
        logger.info(f"Requesting page {page} for country={country}")
        resp = request_with_retries(BASE_URL, params=params, headers=headers)
        if resp is None:
            logger.error(f"Network/client error fetching page {page}. Stopping.")
            break

        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON for page {page}: {e}")
            break

        recordings = data.get("recordings", [])
        if not recordings:
            logger.info(f"No recordings on page {page} -> finished paging.")
            break

        logger.info(f"Page {page}: got {len(recordings)} recordings. Filtering for birds.")
        total_fetched += len(recordings)

        # filter bird records
        for rec in recordings:
            if looks_like_bird(rec):
                for drop in ("sono", "osci"):
                    if drop in rec:
                        rec.pop(drop, None)
                all_bird_records.append(rec)

        # check pagination
        num_pages = None
        for k in ("num_pages", "numPages", "numPagesTotal", "num_pages_total"):
            if k in data:
                try:
                    num_pages = int(data[k])
                    break
                except Exception:
                    continue
        if num_pages and page >= num_pages:
            logger.info(f"Reached num_pages={num_pages}. Done.")
            break

        page += 1

    logger.info(f"Finished fetching. Total records scanned: {total_fetched}. Bird records kept: {len(all_bird_records)}.")
    return all_bird_records


def save_records(records: List[Dict], out_csv: str) -> None:
    """Save list of dicts to CSV."""
    if not records:
        logger.warning("No records to save.")
        return
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Xeno-Canto bird metadata for a country")
    parser.add_argument("--country", choices=["Malaysia", "Singapore", "Brunei"], default="Malaysia",
                        help="Country to fetch bird records for")
    parser.add_argument("--out-csv", default=OUT_CSV_DEFAULT,
                        help="Output CSV filename")
    args = parser.parse_args()

    api_key = os.environ.get("XENO_API_KEY")
    if not api_key:
        logger.error("No XENO_API_KEY found. Set environment variable XENO_API_KEY and retry.")
        sys.exit(2)

    logger.info(f"Starting Xeno-Canto {args.country} -> bird records fetch")
    records = fetch_country_birds(api_key=api_key, country=args.country)
    save_records(records, args.out_csv)
    logger.info("Done.")


if __name__ == "__main__":
    main()
