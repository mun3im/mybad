#!/usr/bin/env python3
"""
Stage 3: Extract negative samples from FSC22 dataset
- Extracts 3s clips from center of audio files
- Filters out bird sounds (positive samples)
- Applies quality filtering (RMS threshold, zero-check)
- Target sample rate: 16kHz
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import logging
from tqdm import tqdm
from pathlib import Path

# ============================= CONFIGURATION =============================
AUDIO_DIR = "/Volumes/Evo/datasets/FSC22/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
METADATA_PATH = "/Volumes/Evo/datasets/FSC22/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"
POS_DIR = "/Volumes/Evo/mybad2/positive/fsc"  # Bird sounds (BirdChirping + WingFlaping)
NEG_DIR = "/Volumes/Evo/mybad2/negative/fsc"  # Non-bird sounds

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples
QUIET_RMS = 0.0001  # Minimum RMS threshold (uniform across all stages)

# Bird-related classes (positive samples - will be skipped)
BIRD_CLASSES = {23, 24}  # BirdChirping & WingFlaping

# ============================= SETUP =============================
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# Setup logging (uniform format across all stages)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Stage3_rejections.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= FUNCTIONS =============================

def compute_rms(y: np.ndarray) -> float:
    """Compute RMS of signal y (float numpy array)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def process_audio_file(row, audio_dir, output_dir, quiet_rms=QUIET_RMS):
    """
    Process a single audio file.
    Returns: (success, skip_reason)
    """
    dataset_filename = row['Dataset File Name']
    class_id = row['Class ID']

    # Skip bird classes (positive samples)
    if class_id in BIRD_CLASSES:
        return True, 'bird_class'

    input_path = os.path.join(audio_dir, dataset_filename)

    if not os.path.exists(input_path):
        logger.error(f"Missing source file: {input_path}")
        return False, 'missing_file'

    try:
        # Load at 16 kHz directly
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        # Check minimum length
        if len(y) < CLIP_SAMPLES:
            logger.warning(f"Safety check failed for {dataset_filename}: too short ({len(y)/TARGET_SR:.2f}s < {CLIP_DURATION}s)")
            return False, 'too_short'

        # Extract 3-second center segment
        center = len(y) // 2
        start = center - CLIP_SAMPLES // 2
        clip = y[start : start + CLIP_SAMPLES]

        # Safety check: all-zero clip
        if np.all(clip == 0):
            logger.warning(f"Safety check failed for {dataset_filename}: clip contains all zeros")
            return False, 'all_zero'

        # Quality check: RMS threshold
        rms = compute_rms(clip)
        if rms < quiet_rms:
            logger.warning(f"Safety check failed for {dataset_filename}: RMS {rms:.6f} < {quiet_rms}")
            return False, 'too_quiet'

        # Save negative sample
        base_filename = os.path.basename(dataset_filename)
        new_filename = f"fsc-{base_filename}"
        output_path = os.path.join(output_dir, new_filename)
        sf.write(output_path, clip, TARGET_SR)
        
        return True, None

    except Exception as e:
        logger.error(f"Exception processing {dataset_filename} (class_id: {class_id}): {type(e).__name__}: {e}")
        return False, 'processing_error'


# ============================= MAIN PROCESSING =============================

def main():
    logger.info("=" * 70)
    logger.info("STAGE 3: FSC22 → NEGATIVE SAMPLES EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Source:  {AUDIO_DIR}")
    logger.info(f"Output:  {NEG_DIR}")
    logger.info(f"Config:  {CLIP_DURATION}s clips @ {TARGET_SR} Hz, RMS threshold = {QUIET_RMS}")
    logger.info("")

    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    total_files = len(metadata)

    # Statistics
    stats = {
        'negative_saved': 0,
        'bird_class': 0,
        'too_short': 0,
        'all_zero': 0,
        'too_quiet': 0,
        'missing_file': 0,
        'processing_error': 0
    }

    # Process all files
    for _, row in tqdm(metadata.iterrows(), total=total_files, desc="Processing FSC22"):
        success, reason = process_audio_file(row, AUDIO_DIR, NEG_DIR, QUIET_RMS)

        if success:
            if reason == 'bird_class':
                stats['bird_class'] += 1
            else:
                stats['negative_saved'] += 1
        else:
            stats[reason] = stats.get(reason, 0) + 1

    # ============================= SUMMARY =============================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 3 COMPLETE - FSC22 EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files processed:              {total_files:5d}")
    logger.info(f"Negative samples saved:             {stats['negative_saved']:5d}")
    logger.info(f"Skipped (bird classes):             {stats['bird_class']:5d}")
    logger.info(f"Skipped (shorter than {CLIP_DURATION}s):       {stats['too_short']:5d}")
    logger.info(f"Skipped (all-zero audio):           {stats['all_zero']:5d}")
    logger.info(f"Skipped (RMS < {QUIET_RMS}):         {stats['too_quiet']:5d}")
    logger.info(f"Errors (missing files):             {stats['missing_file']:5d}")
    logger.info(f"Errors (processing):                {stats['processing_error']:5d}")
    logger.info("")
    logger.info(f"Output directory: {NEG_DIR}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())