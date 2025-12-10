import os
import sys
import pandas as pd
from pydub import AudioSegment
import librosa
import soundfile as sf
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    filename='audio_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
csv_path = '/Volumes/Evo/datasets/ESC50/meta/esc50.csv'
input_dir = '/Volumes/Evo/datasets/ESC50/audio'
neg_dir = '/Volumes/Evo/mybad2/negative/esc'
pos_dir = '/Volumes/Evo/mybad2/positive/esc'

# Create output directories if needed
os.makedirs(neg_dir, exist_ok=True)
os.makedirs(pos_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Define bird categories that should go to positive directory
bird_categories = ['crow', 'hen', 'rooster', 'chirping_birds']

# Initialize counters
positive_count = 0
negative_count = 0
skipped_short = 0

# Target sample rate and clip duration
TARGET_SR = 16000
CLIP_DURATION = 3.0  # seconds
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)  # 48000 samples

for idx, row in df.iterrows():
    filename = row['filename']
    category = row['category']
    input_path = os.path.join(input_dir, filename)

    # Determine output directory based on category
    if category in bird_categories:
        output_dir = pos_dir
    else:
        output_dir = neg_dir

    if not os.path.isfile(input_path):
        logging.warning(f"File not found: {input_path}")
        print(f"File not found: {input_path}")
        continue

    try:
        # Load and resample to 16000 Hz
        y, sr = librosa.load(input_path, sr=TARGET_SR)

        # Calculate duration in seconds
        duration = len(y) / TARGET_SR

        # Skip files shorter than 3 seconds
        if duration < CLIP_DURATION:
            logging.info(f"Skipped (< 3s): {filename} (duration: {duration:.2f}s)")
            print(f"Skipped (< 3s): {filename} (duration: {duration:.2f}s)")
            skipped_short += 1
            continue

        # Extract exact 3-second clip from center
        center_sample = len(y) // 2
        start_sample = center_sample - (CLIP_SAMPLES // 2)
        end_sample = start_sample + CLIP_SAMPLES

        center_clip = y[start_sample:end_sample]

        # Verify we got exactly 3 seconds (48000 samples at 16kHz)
        if len(center_clip) != CLIP_SAMPLES:
            logging.warning(f"Clip length mismatch for {filename}: {len(center_clip)} samples")
            continue

        # Check if the clip contains all zeros
        if np.all(center_clip == 0):
            logging.info(f"Zero clip detected: {filename}")
            skipped_short += 1
            print(f"Zero clip detected and skipped: {filename}")
            continue

        # Save as temporary WAV for pydub processing
        temp_wav = "/tmp/temp_center_clip.wav"
        sf.write(temp_wav, center_clip, TARGET_SR)

        audio = AudioSegment.from_wav(temp_wav)
        # Ensure mono output
        if audio.channels > 1:
            audio = audio.set_channels(1)

        out_filename = f"esc-{os.path.splitext(filename)[0]}.wav"
        out_path = os.path.join(output_dir, out_filename)
        audio.export(out_path, format="wav")

        logging.info(f"Saved: {out_path} (category: {category}, 3s center clip @ 16kHz)")
        print(f"Saved: {out_path} (category: {category})")

        # Update counters
        if category in bird_categories:
            positive_count += 1
        else:
            negative_count += 1

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        print(f"Error processing {filename}: {e}")

print(f"\nProcessing complete!")
print(f"Positive samples (bird sounds): {positive_count}")
print(f"Negative samples (non-bird sounds): {negative_count}")
print(f"Files skipped (< 3s or zero): {skipped_short}")
print(f"Total samples saved: {positive_count + negative_count}")
print(f"All clips are exactly 3 seconds at 16kHz")