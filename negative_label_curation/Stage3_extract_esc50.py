#!/usr/bin/env python3
"""
Stage: Extract negative samples from ESC-50 dataset
- First pass: collect all candidate clips from selected environmental categories with their RMS
- Sort by RMS descending
- Keep only the top 444 loudest (after quality filters)
- Extracts 3s center clips, saves to NEG_DIR
- Target sample rate: 16kHz
"""

import os
import sys
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import logging

# ============================= CONFIGURATION =============================
CSV_PATH = '/Volumes/Evo/datasets/ESC50/meta/esc50.csv'
AUDIO_DIR = '/Volumes/Evo/datasets/ESC50/audio/'
NEG_DIR = '/Volumes/Evo/seabad/negative/esc'  # Top 444 loudest selected environmental sounds

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples
QUIET_RMS = 0.00003  # Minimum RMS threshold (still applied)
MAX_SAMPLES_TO_KEEP = 444

# Selected environmental categories to consider (exact names from ESC-50)
ENVIRONMENTAL_CATEGORIES = {
    'rain',
    'sea_waves',
    'crackling_fire',
    'crickets',
    'water_drops',
    'thunderstorm',
    'helicopter',
    'chainsaw',
    'car_horn',
    'engine',
    'airplane',
    'wind', 
}

# ============================= SETUP =============================
os.makedirs(NEG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ESC50_extraction_top444.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================= FUNCTIONS =============================

def compute_rms(y: np.ndarray) -> float:
    """Compute RMS of signal y."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))

def extract_clip_and_rms(row, audio_dir):
    """
    Extract the 3s center clip and compute its RMS.
    Returns (clip: np.ndarray or None, rms: float, reason: str or None)
    """
    filename = row['filename']
    src_path = os.path.join(audio_dir, filename)

    if not os.path.isfile(src_path):
        return None, 0.0, 'missing_file'

    try:
        y, _ = librosa.load(src_path, sr=TARGET_SR, mono=True)

        center = len(y) // 2
        clip = y[center - CLIP_SAMPLES // 2 : center + CLIP_SAMPLES // 2]

        if len(clip) != CLIP_SAMPLES:
            return None, 0.0, 'invalid_length'

        if np.all(clip == 0):
            return None, 0.0, 'all_zero'

        rms = compute_rms(clip)
        if rms < QUIET_RMS:
            return None, rms, 'too_quiet'

        return clip, rms, None

    except Exception as e:
        logger.error(f"Error loading {filename}: {type(e).__name__}: {e}")
        return None, 0.0, 'processing_error'

# ============================= MAIN =============================

def main():
    logger.info("=" * 70)
    logger.info("ESC-50 → TOP 444 LOUDEST SELECTED ENVIRONMENTAL NEGATIVES")
    logger.info("=" * 70)
    logger.info(f"Source:  {AUDIO_DIR}")
    logger.info(f"Output:  {NEG_DIR}")
    logger.info(f"Selected categories: {', '.join(sorted(ENVIRONMENTAL_CATEGORIES))}")
    logger.info(f"Max to keep: {MAX_SAMPLES_TO_KEEP} loudest")
    logger.info("")

    if not os.path.exists(CSV_PATH):
        logger.error(f"Metadata CSV not found: {CSV_PATH}")
        return 1

    df = pd.read_csv(CSV_PATH)

    # Filter to selected categories
    candidates_df = df[df['category'].isin(ENVIRONMENTAL_CATEGORIES)].copy()
    logger.info(f"Candidate files in selected categories: {len(candidates_df)}")

    # First pass: collect valid clips with RMS
    candidates = []
    skip_reasons = {'missing_file': 0, 'invalid_length': 0, 'all_zero': 0, 'too_quiet': 0, 'processing_error': 0}

    for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Pass 1: Compute RMS"):
        category = row['category']
        filename = row['filename']

        clip, rms, reason = extract_clip_and_rms(row, AUDIO_DIR)

        if clip is not None:
            candidates.append({
                'clip': clip,
                'rms': rms,
                'filename': filename,
                'category': category
            })
        else:
            # Track skip reasons
            if reason:
                skip_reasons[reason] += 1
                logger.debug(f"Skipped {filename} ({category}): {reason} (RMS: {rms:.6f})")

    logger.info(f"Valid candidates after quality filters: {len(candidates)}")
    logger.info(f"Files filtered out by reason: {dict(skip_reasons)}")

    if len(candidates) == 0:
        logger.error("No valid candidates found!")
        return 1

    # Sort by RMS descending (loudest first)
    candidates.sort(key=lambda x: x['rms'], reverse=True)

    # Keep top N
    to_save = candidates[:MAX_SAMPLES_TO_KEEP]
    logger.info(f"Keeping top {len(to_save)} loudest samples (target: {MAX_SAMPLES_TO_KEEP})")

    # Log RMS range of what we're keeping
    if to_save:
        logger.info(f"RMS range of kept samples: {to_save[-1]['rms']:.6f} to {to_save[0]['rms']:.6f}")

    # Second pass: save the selected clips
    saved_count = 0
    for item in tqdm(to_save, desc="Pass 2: Saving files"):
        new_name = f"esc-{os.path.splitext(item['filename'])[0]}.wav"
        out_path = os.path.join(NEG_DIR, new_name)
        sf.write(out_path, item['clip'], TARGET_SR)
        saved_count += 1

    # ============================= SUMMARY =============================
    logger.info("")
    logger.info("=" * 70)
    logger.info("ESC-50 TOP 444 EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files in selected categories:        {len(candidates_df):5d}")
    logger.info(f"Valid after filters:                       {len(candidates):5d}")
    logger.info(f"Negative samples saved (top loudest):       {saved_count:5d}")
    logger.info(f"Loudest RMS:                               {to_save[0]['rms']:.6f}" if to_save else "N/A")
    logger.info(f"Quietest kept RMS:                         {to_save[-1]['rms']:.6f}" if to_save else "N/A")
    logger.info("")
    logger.info(f"Output → {NEG_DIR}")
    logger.info("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())