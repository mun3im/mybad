#!/usr/bin/env python3
"""
Stage 4: Extract and filter DATASEC dataset
- Extracts 3s clips from center of audio files
- Separates bird sounds (Crows*, Chicken*, Birds*) from non-bird sounds
- Filters quality and selects exactly 3637 non-bird samples
"""

import os
import glob
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# ============================= CONFIG =============================
DATASEC_ROOT = '/Volumes/Evo/datasets/DataSEC'
POS_DIR = '/Volumes/Evo/mybad2/positive/datasec'  # bird sounds
NEG_DIR = '/Volumes/Evo/mybad2/negative/datasec'  # non-bird sounds

# Bird-related folder patterns
BIRD_PATTERNS = ['Crows', 'Chicken', 'Birds']

TARGET_SR = 16000
CLIP_DURATION = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000

# Quality thresholds for filtering (uniform across all stages)
QUIET_RMS = 0.0001       # Minimum RMS energy (uniform with other stages)
MAX_PEAK = 0.98          # Maximum peak (clipping threshold)
MIN_DYNAMIC_RANGE = 0.1  # Minimum dynamic range (max - min)

# Target: exactly 3637 non-bird samples
TARGET_NEGATIVE_TOTAL = 3654

# Create output directories
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# Setup error logging (uniform format across all stages)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datasec_processing.log'),
        logging.FileHandler('Stage4_rejections.txt', mode='w')  # Uniform format across all stages
    ]
)

# ============================= FUNCTIONS =============================

def is_bird_folder(folder_name):
    """Check if folder contains bird sounds."""
    for pattern in BIRD_PATTERNS:
        if folder_name.startswith(pattern):
            return True
    return False

def get_all_folders():
    """Get all folders in DATASEC root."""
    all_folders = [f for f in os.listdir(DATASEC_ROOT)
                   if os.path.isdir(os.path.join(DATASEC_ROOT, f))]
    return sorted(all_folders)

def get_audio_files(folder_path):
    """Get all audio files (wav, mp3, flac, ogg) from a folder."""
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.WAV', '*.MP3', '*.FLAC', '*.OGG']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        # Also check subdirectories
        audio_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))

    return audio_files

def analyze_audio_quality(y):
    """
    Analyze audio quality from numpy array.
    Returns: (is_good, rms, peak, dynamic_range)
    """
    # Calculate metrics
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    dynamic_range = float(np.max(y) - np.min(y))

    # Check quality criteria (using uniform QUIET_RMS threshold)
    is_good = (
        rms >= QUIET_RMS and
        peak <= MAX_PEAK and
        dynamic_range >= MIN_DYNAMIC_RANGE
    )

    return is_good, rms, peak, dynamic_range

def calculate_spectral_diversity_score(y, sr):
    """
    Calculate a diversity score based on spectral features.
    Higher score = more diverse/interesting sample.
    """
    try:
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Combine into diversity score (normalized)
        score = (
            spectral_centroid / 8000 +      # normalized by Nyquist/2
            spectral_rolloff / 8000 +
            spectral_bandwidth / 4000 +
            zcr * 10
        )

        return float(score)

    except Exception as e:
        return 0.0

# ============================= PHASE 1: EXTRACT ALL CLIPS =============================

print("=" * 70)
print("DATASEC EXTRACTION - PHASE 1: Extract all 3-second clips")
print("=" * 70)
print(f"Source: {DATASEC_ROOT}")
print(f"Positive (bird sounds) → {POS_DIR}")
print(f"Negative (non-bird)    → {NEG_DIR}\n")

# Get all folders
folders = get_all_folders()
bird_folders = [f for f in folders if is_bird_folder(f)]
non_bird_folders = [f for f in folders if not is_bird_folder(f)]

print(f"Found {len(folders)} total folders:")
print(f"  Bird folders ({len(bird_folders)}): {', '.join(bird_folders)}")
print(f"  Non-bird folders ({len(non_bird_folders)}): {', '.join(non_bird_folders)}")
print()

# Temporary storage for negative samples with metadata
negative_samples = []  # List of (folder, clip_array, diversity_score, output_filename)

# Stats
positive_count = 0
skipped_short = 0
skipped_zero = 0
errors = 0
file_counter = 0

# Process each folder
for folder in folders:
    folder_path = os.path.join(DATASEC_ROOT, folder)
    audio_files = get_audio_files(folder_path)

    if not audio_files:
        print(f"No audio files found in {folder}")
        continue

    print(f"\nProcessing folder: {folder} ({len(audio_files)} files)")
    is_bird = is_bird_folder(folder)

    for audio_file in tqdm(audio_files, desc=f"  {folder}"):
        try:
            # Load at 16 kHz directly
            y, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)

            # Skip if shorter than 3 seconds
            if len(y) < CLIP_SAMPLES:
                skipped_short += 1
                error_msg = f"Safety check failed: too short ({len(y)/TARGET_SR:.2f}s < {CLIP_DURATION}s) - {audio_file}"
                logging.warning(error_msg)
                continue

            # Extract 3-second center segment
            center = len(y) // 2
            start = center - CLIP_SAMPLES // 2
            clip = y[start : start + CLIP_SAMPLES]

            # Safety check: all-zero clip?
            if np.all(clip == 0):
                skipped_zero += 1
                error_msg = f"Safety check failed: clip contains all zeros - {audio_file}"
                logging.warning(error_msg)
                continue

            # Create output filename
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            folder_clean = folder.replace(' ', '_').replace('/', '_')
            output_filename = f"datasec-{file_counter:05d}-{folder_clean}-{basename}.wav"
            file_counter += 1

            if is_bird:
                # Save positive samples directly
                output_path = os.path.join(POS_DIR, output_filename)
                sf.write(output_path, clip, TARGET_SR)
                positive_count += 1
            else:
                # Store negative samples for filtering
                negative_samples.append((folder, clip, None, output_filename))  # score calculated later

        except Exception as e:
            errors += 1
            logging.error(f"Error processing {audio_file}: {e}")

print(f"\n{'=' * 70}")
print("PHASE 1 COMPLETE - Extraction Summary")
print(f"{'=' * 70}")
print(f"Positive (bird) clips saved:  {positive_count:4d}")
print(f"Negative (non-bird) extracted: {len(negative_samples):4d}")
print(f"Skipped (shorter than 3s):    {skipped_short:4d}")
print(f"Skipped (all-zero audio):     {skipped_zero:4d}")
print(f"Errors:                       {errors:4d}")

# ============================= PHASE 2: FILTER NEGATIVE SAMPLES =============================

print(f"\n{'=' * 70}")
print(f"DATASEC FILTERING - PHASE 2: Filter to exactly {TARGET_NEGATIVE_TOTAL} negative samples")
print(f"{'=' * 70}")

# Step 1: Quality filter
print("\nStep 1: Quality filtering...")
quality_stats = {
    'too_quiet': 0,
    'clipped': 0,
    'low_dynamic_range': 0,
    'passed': 0
}

# Group by folder and filter
folder_groups = {}
for folder, clip, _, filename in tqdm(negative_samples, desc="Quality filtering"):
    is_good, rms, peak, dynamic_range = analyze_audio_quality(clip)

    if is_good:
        if folder not in folder_groups:
            folder_groups[folder] = []
        folder_groups[folder].append((clip, filename))
        quality_stats['passed'] += 1
    else:
        if rms < QUIET_RMS:
            quality_stats['too_quiet'] += 1
            logging.warning(f"Quality check failed: too quiet (RMS {rms:.6f} < {QUIET_RMS})")
        if peak > MAX_PEAK:
            quality_stats['clipped'] += 1
            logging.warning(f"Quality check failed: clipped (peak {peak:.4f} > {MAX_PEAK})")
        if dynamic_range < MIN_DYNAMIC_RANGE:
            quality_stats['low_dynamic_range'] += 1
            logging.warning(f"Quality check failed: low dynamic range ({dynamic_range:.4f} < {MIN_DYNAMIC_RANGE})")

print(f"\nQuality filtering results:")
print(f"  Passed:              {quality_stats['passed']:4d}")
print(f"  Too quiet:           {quality_stats['too_quiet']:4d}")
print(f"  Clipped:             {quality_stats['clipped']:4d}")
print(f"  Low dynamic range:   {quality_stats['low_dynamic_range']:4d}")

# Step 2: Calculate allocation
print(f"\nStep 2: Calculating allocation for exactly {TARGET_NEGATIVE_TOTAL} samples...")

# Identify Music and Voices folders
music_folder = None
voices_folder = None
other_folders = {}

for folder, clips in folder_groups.items():
    folder_lower = folder.lower()
    if 'music' in folder_lower:
        music_folder = (folder, clips)
    elif 'voice' in folder_lower or 'speech' in folder_lower:
        voices_folder = (folder, clips)
    else:
        other_folders[folder] = clips

# Count samples from other folders (we'll keep all of these)
other_count = sum(len(clips) for clips in other_folders.values())
print(f"\nSmall folders (keep all after quality filter): {other_count} samples")

if music_folder is None or voices_folder is None:
    print("WARNING: Could not identify Music or Voices folders!")
    print("Available folders:", list(folder_groups.keys()))
    # Just take first TARGET_NEGATIVE_TOTAL samples
    final_selection = {}
    count = 0
    for folder, clips in folder_groups.items():
        final_selection[folder] = []
        for clip, filename in clips:
            if count >= TARGET_NEGATIVE_TOTAL:
                break
            final_selection[folder].append((clip, filename))
            count += 1
        if count >= TARGET_NEGATIVE_TOTAL:
            break
else:
    music_name, music_clips = music_folder
    voices_name, voices_clips = voices_folder

    print(f"Music folder:  {len(music_clips)} samples available")
    print(f"Voices folder: {len(voices_clips)} samples available")

    # Calculate how many we need from Music and Voices
    remaining = TARGET_NEGATIVE_TOTAL - other_count
    print(f"\nNeed to select {remaining} samples from Music + Voices")

    # Allocate with bias AGAINST music (unlikely in jungle/field deployment)
    # Use 1:3 ratio (Music:Voices) instead of proportional
    # This prioritizes voices/speech which is more likely in field settings
    music_allocation = int(remaining * 0.05)  # 5% for music
    voices_allocation = remaining - music_allocation  # 95% for voices

    # Ensure we don't exceed available
    music_allocation = min(music_allocation, len(music_clips))
    voices_allocation = min(voices_allocation, len(voices_clips))

    # Adjust if total doesn't match (due to rounding)
    total_allocated = music_allocation + voices_allocation
    if total_allocated < remaining:
        # Add remainder to the larger folder
        if len(music_clips) > len(voices_clips):
            music_allocation += remaining - total_allocated
        else:
            voices_allocation += remaining - total_allocated

    print(f"\nFinal allocation:")
    print(f"  Music:  {music_allocation} samples")
    print(f"  Voices: {voices_allocation} samples")
    print(f"  Others: {other_count} samples")
    print(f"  TOTAL:  {music_allocation + voices_allocation + other_count} samples")

    # Step 3: Select samples using diversity scoring
    print("\nStep 3: Selecting samples by diversity...")
    final_selection = {}

    # Keep all from other folders
    final_selection.update(other_folders)

    # Select from Music folder
    print(f"  Scoring and selecting from {music_name}...")
    music_scores = []
    for clip, filename in tqdm(music_clips, desc=f"    Scoring {music_name}", leave=False):
        score = calculate_spectral_diversity_score(clip, TARGET_SR)
        music_scores.append((clip, filename, score))
    music_scores.sort(key=lambda x: x[2], reverse=True)
    final_selection[music_name] = [(clip, filename) for clip, filename, score in music_scores[:music_allocation]]

    # Select from Voices folder
    print(f"  Scoring and selecting from {voices_name}...")
    voices_scores = []
    for clip, filename in tqdm(voices_clips, desc=f"    Scoring {voices_name}", leave=False):
        score = calculate_spectral_diversity_score(clip, TARGET_SR)
        voices_scores.append((clip, filename, score))
    voices_scores.sort(key=lambda x: x[2], reverse=True)
    final_selection[voices_name] = [(clip, filename) for clip, filename, score in voices_scores[:voices_allocation]]

# Step 4: Save selected negative samples
print("\nStep 4: Saving selected negative samples...")
total_saved = 0

for folder, clips in final_selection.items():
    for clip, filename in tqdm(clips, desc=f"  Saving {folder}", leave=False):
        output_path = os.path.join(NEG_DIR, filename)
        sf.write(output_path, clip, TARGET_SR)
        total_saved += 1

# ============================= FINAL SUMMARY =============================
print(f"\n{'=' * 70}")
print("DATASEC EXTRACTION AND FILTERING COMPLETE")
print(f"{'=' * 70}")
print(f"Positive (bird sounds):       {positive_count:4d}")
print(f"Negative (non-bird sounds):   {total_saved:4d}")
print(f"Negative target:              {TARGET_NEGATIVE_TOTAL:4d}")

if total_saved == TARGET_NEGATIVE_TOTAL:
    print(f"\n✓ SUCCESS: Exactly {TARGET_NEGATIVE_TOTAL} negative samples selected!")
else:
    print(f"\n⚠ WARNING: Got {total_saved} samples, target was {TARGET_NEGATIVE_TOTAL}")

print(f"\nNegative samples per folder:")
for folder, clips in sorted(final_selection.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {folder:30s}: {len(clips):4d} files")

print(f"\nOutput folders:")
print(f"   Positive → {POS_DIR}")
print(f"   Negative → {NEG_DIR}")
print(f"{'=' * 70}")

# Log summary
logging.info(f"Processing complete: {positive_count} positive, {total_saved} negative saved")
logging.info(f"Rejected: {skipped_short} too short, {skipped_zero} all-zero, {errors} errors")
logging.info(f"Quality filter: {quality_stats['too_quiet']} too quiet, {quality_stats['clipped']} clipped, {quality_stats['low_dynamic_range']} low dynamic range")
