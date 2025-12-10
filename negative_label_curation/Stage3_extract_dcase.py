#!/usr/bin/env python3
"""
Updated: skip & log files < 3s and RMS < QUIET_RMS (default 0.003).
Writes skipped_files.csv with details and reasons.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dataset configuration
datasets = {
    # "BirdVox-DCASE-20k": {
    #     "csv": Path("~/datasets/birdvox/BirdVoxDCASE20k_csvpublic.csv").expanduser(),
    #     "wav": Path("~/datasets/birdvox/wav").expanduser(),
    #     "subdir": "bv"
    # },
    "Freefield1010": {
        "csv": Path("/Volumes/Evo/datasets/freefield1010/ff1010bird_metadata_2018.csv").expanduser(),
        "wav": Path("/Volumes/Evo/datasets/freefield1010/wav").expanduser(),
        "subdir": "ff"
    },
    "Warblrb10k": {
        "csv": Path("/Volumes/Evo/datasets/warblr/warblrb10k_public_metadata_2018.csv").expanduser(),
        "wav": Path("/Volumes/Evo/datasets/warblr/wav").expanduser(),
        "subdir": "wb"
    },
}

# Base output directory
base_output_dir = Path("/Volumes/Evo/mybad2/negative").expanduser()

# Audio processing parameters
TARGET_SR = 16000
CLIP_DURATION = 3.0
MIN_DURATION = 3.0  # Skip files shorter than this
SAMPLES_PER_CLIP = int(TARGET_SR * CLIP_DURATION)

# Quiet threshold (RMS). Reasonable range ~ 0.001 - 0.005. 0.003 is a good middle value.
QUIET_RMS = 0.001

# CSV for skipped files
SKIPPED_CSV = Path("skipped_files.csv")


def compute_rms(y: np.ndarray) -> float:
    """Compute RMS of signal y (float numpy array)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def process_audio_file(row_dict, dataset_info, output_dir, quiet_rms=QUIET_RMS):
    """
    row_dict: dict-like with at least 'itemid'
    Returns tuple (success_bool, skipped_info_or_outpath)
    If success_bool is True -> out_path string returned
    If False -> dict describing skip: {'reason': 'short'|'low_rms'|'missing'|'error', 'duration':..., 'rms':...}
    """
    try:
        itemid = row_dict.get("itemid")
        filename = f"{itemid}.wav"
        filepath = dataset_info["wav"] / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return False, {"reason": "missing", "filepath": str(filepath)}

        # Load and resample audio to TARGET_SR, mono
        y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)  # y in float32 [-1,1]

        audio_duration = float(len(y) / TARGET_SR)
        rms_full = compute_rms(y)

        # Skip if too short
        if audio_duration < MIN_DURATION:
            return False, {"reason": "short", "duration": audio_duration, "rms": rms_full, "filepath": str(filepath)}

        # Skip if very low RMS
        if rms_full < quiet_rms:
            return False, {"reason": "low_rms", "duration": audio_duration, "rms": rms_full, "filepath": str(filepath)}

        # Extract 3s clip from center
        total_samples = len(y)
        start_sample = max(0, (total_samples - SAMPLES_PER_CLIP) // 2)
        clip = y[start_sample:start_sample + SAMPLES_PER_CLIP]
        # safety: fix length
        clip = librosa.util.fix_length(clip, size=SAMPLES_PER_CLIP)

        # Convert to int16 and save using pydub (as in your original code)
        clip_int16 = np.int16(np.clip(clip, -1.0, 1.0) * 32767)
        audio = AudioSegment(
            clip_int16.tobytes(),
            frame_rate=TARGET_SR,
            sample_width=2,
            channels=1
        )

        out_filename = f"{dataset_info['subdir']}-{itemid}.wav"
        out_path = output_dir / out_filename
        audio.export(out_path, format="wav")

        return True, str(out_path)

    except Exception as e:
        logger.exception(f"Error processing {row_dict.get('itemid')}: {e}")
        return False, {"reason": "error", "error": str(e)}


def process_dataset(dataset_name, dataset_info, quiet_rms=QUIET_RMS):
    """Process all files in a dataset and log skipped items."""
    logger.info(f"Processing dataset: {dataset_name}")

    output_dir = base_output_dir / dataset_info["subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    try:
        df = pd.read_csv(dataset_info["csv"])
        no_bird_df = df[df.get("hasbird", 0) == 0]  # handle if column missing
        total_files = len(no_bird_df)
        logger.info(f"Found {total_files} negative files (hasbird == 0)")

        # Prepare skipped CSV (append if exists)
        skipped_rows = []

        successful = 0
        skipped = 0

        # convert rows to dicts for safer pickling in ProcessPool
        row_dicts = [row[1].to_dict() for row in no_bird_df.iterrows()]

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_audio_file, row_dict, dataset_info, output_dir, quiet_rms)
                for row_dict in row_dicts
            ]

            with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="files") as pbar:
                for future in as_completed(futures):
                    ok, info = future.result()
                    if ok:
                        successful += 1
                    else:
                        skipped += 1
                        # normalize info dict
                        rec = {
                            "dataset": dataset_name,
                            "itemid": info.get("itemid", None),
                            "filepath": info.get("filepath", None),
                            "duration": info.get("duration", None),
                            "rms": info.get("rms", None),
                            "reason": info.get("reason", "unknown"),
                        }
                        skipped_rows.append(rec)
                    pbar.update(1)

        logger.info(f"Dataset {dataset_name} - Successfully processed: {successful}/{total_files}")
        logger.info(f"Dataset {dataset_name} - Skipped/Failed: {skipped}/{total_files}")

        # append skipped_rows to SKIPPED_CSV
        if skipped_rows:
            write_header = not SKIPPED_CSV.exists()
            with SKIPPED_CSV.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "itemid", "filepath", "duration", "rms", "reason"])
                if write_header:
                    writer.writeheader()
                for r in skipped_rows:
                    writer.writerow(r)
            logger.info(f"Wrote {len(skipped_rows)} rows to {SKIPPED_CSV}")

    except Exception as e:
        logger.exception(f"Could not process dataset {dataset_name}: {e}")


def main():
    base_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {CLIP_DURATION}s clips at {TARGET_SR} Hz from center of audio files")
    logger.info(f"Files shorter than {MIN_DURATION}s or RMS < {QUIET_RMS} will be skipped")
    logger.info("=" * 50)

    # Process each dataset
    for dataset_name, dataset_info in datasets.items():
        process_dataset(dataset_name, dataset_info, quiet_rms=QUIET_RMS)
        logger.info("=" * 50)

    logger.info(f"\nProcessing complete! Files saved to subdirectories under: {base_output_dir}")
    for dataset_name, dataset_info in datasets.items():
        logger.info(f"  - {dataset_name} → {dataset_info['subdir']}/")


if __name__ == "__main__":
    main()
