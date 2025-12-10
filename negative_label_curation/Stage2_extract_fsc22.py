import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import logging
from tqdm import tqdm  # nice progress bar

# ============================= CONFIG =============================
audio_dir       = "/Volumes/Evo/datasets/FSC22/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
metadata_path   = "/Volumes/Evo/datasets/FSC22/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"

positive_dir    = "/Volumes/Evo/mybad2/positive/fsc"   # BirdChirping (23) + WingFlaping (24)
negative_dir    = "/Volumes/Evo/mybad2/negative/fsc"   # Everything else

TARGET_SR       = 16000
CLIP_DURATION   = 3.0                     # seconds
CLIP_SAMPLES    = int(TARGET_SR * CLIP_DURATION)

# Create folders
os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)

# Logging
logging.basicConfig(
    filename='fsc22_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Bird-related classes
BIRD_CLASSES = {23, 24}   # BirdChirping & WingFlaping

# Stats
positive_count = 0
negative_count = 0
skipped_short  = 0
skipped_zero   = 0
errors         = 0

print("Starting FSC22 → 3-second bird / non-bird split...\n")

for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
    dataset_filename = row['Dataset File Name']
    class_id         = row['Class ID']

    input_path = os.path.join(audio_dir, dataset_filename)

    if not os.path.exists(input_path):
        logging.warning(f"File not found: {input_path}")
        errors += 1
        continue

    try:
        # Load at 16 kHz directly
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        if len(y) < CLIP_SAMPLES:
            skipped_short += 1
            logging.info(f"Too short (<3s): {dataset_filename} ({len(y)/TARGET_SR:.2f}s)")
            continue

        # Take exact 3-second center segment
        center = len(y) // 2
        start  = center - CLIP_SAMPLES // 2
        # safe because we already checked length
        clip   = y[start : start + CLIP_SAMPLES]

        # Safety check: all-zero clip?
        if np.all(clip == 0):
            skipped_zero += 1
            logging.info(f"All-zero clip: {dataset_filename}")
            continue

        # Decide positive / negative
        is_bird = class_id in BIRD_CLASSES
        target_dir = positive_dir if is_bird else negative_dir
        counter = positive_count if is_bird else negative_count

        # Filename: keep original name (already unique)
        output_path = os.path.join(target_dir, dataset_filename)

        # Save directly with soundfile (fast, no pydub needed)
        sf.write(output_path, clip, TARGET_SR)

        # Update counter
        if is_bird:
            positive_count += 1
        else:
            negative_count += 1

    except Exception as e:
        logging.error(f"Error processing {dataset_filename}: {e}")
        errors += 1

# ============================= SUMMARY =============================
print("\n" + "="*50)
print("FSC22 → 3-second Bird / Non-Bird Split Finished")
print("="*50)
print(f"Positive (Bird sounds)      – classes 23 & 24) : {positive_count:4d}")
print(f"Negative (Everything else)                       : {negative_count:4d}")
print(f"Skipped (shorter than 3s)                       : {skipped_short:4d}")
print(f"Skipped (all-zero audio)                        : {skipped_zero:4d}")
print(f"Errors / missing files                          : {errors:4d}")
print(f"Total processed & saved                         : {positive_count + negative_count:4d}")
print(f"Output → positive: {positive_dir}")
print(f"         negative: {negative_dir}")
