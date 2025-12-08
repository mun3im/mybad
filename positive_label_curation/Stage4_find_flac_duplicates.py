#!/usr/bin/env python3
"""
Detect near-duplicate 3-second audio clips (WAV & FLAC) across subfolders.
Now correctly handles files that live one level down in species/code subdirs.
"""

import argparse
import argparse
import sys
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
from typing import List, Tuple

# ------------------ CONFIG ------------------
TARGET_SR = 16000
TARGET_DURATION = 3.0
N_MELS = 128
HOP_LENGTH = 128
MIN_DURATION = 2.999

MEAN_SIM_THRESHOLD = 0.997
MIN_SIM_THRESHOLD  = 0.985
P5_SIM_THRESHOLD   = 0.992
# --------------------------------------------

def compute_embedding(filepath: Path) -> np.ndarray | None:
    try:
        y, file_sr = librosa.load(filepath, sr=TARGET_SR, mono=True, res_type='kaiser_fast')
    except Exception as e:
        print(f"Failed loading {filepath.name}: {e}", file=sys.stderr)
        return None

    if len(y) / TARGET_SR < MIN_DURATION:
        return None

    # Force exactly 3.00 seconds
    target_samples = int(TARGET_DURATION * TARGET_SR)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')

    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS,
                                         hop_length=HOP_LENGTH, n_fft=512, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-frame L2 normalization → cosine = dot product
    norms = np.linalg.norm(log_mel, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    log_mel = log_mel / norms

    return log_mel.astype(np.float32)


def frame_cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float, float]:
    min_frames = min(emb1.shape[1], emb2.shape[1])
    if min_frames == 0:
        return 0.0, 0.0, 0.0
    sims = np.clip(np.sum(emb1[:, :min_frames] * emb2[:, :min_frames], axis=0), -1.0, 1.0)
    return float(np.mean(sims)), float(np.min(sims)), float(np.percentile(sims, 5))


def find_similar_pairs(filepaths: List[Path]) -> List[Tuple[Path, Path, float]]:
    print(f"\nComputing embeddings for {len(filepaths)} files...")
    embeddings, valid_paths = [], []

    for fp in tqdm(filepaths, desc="Embedding", unit="file"):
        emb = compute_embedding(fp)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(fp)

    print(f"Embedded {len(valid_paths)}/{len(filepaths)} files successfully.")

    if len(valid_paths) < 2:
        return []

    similar_pairs = []
    total = len(valid_paths) * (len(valid_paths)-1) // 2
    print("Comparing pairs...")
    with tqdm(total=total, desc="Comparing", unit="pair") as pbar:
        for i in range(len(valid_paths)):
            for j in range(i+1, len(valid_paths)):
                mean_sim, min_sim, p5_sim = frame_cosine_sim(embeddings[i], embeddings[j])
                if (mean_sim >= MEAN_SIM_THRESHOLD and
                    min_sim  >= MIN_SIM_THRESHOLD and
                    p5_sim   >= P5_SIM_THRESHOLD):
                    similar_pairs.append((valid_paths[i], valid_paths[j], mean_sim))
                pbar.update(1)

    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    return similar_pairs


def collect_audio_files(root: Path, recursive: bool) -> List[Path]:
    """
    Collect all .wav and .flac files.
    Default behaviour: root + immediate subfolders only (most common case)
    --recursive flag → full recursion
    """
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    candidates = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]

    # If we found nothing and not in recursive mode, explicitly check one level deeper
    if not candidates and not recursive:
        candidates = []
        for subdir in root.iterdir():
            if subdir.is_dir():
                candidates.extend([
                    p for p in subdir.glob("*")
                    if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
                ])

    return sorted(candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Find near-duplicate 3-second clips (WAV/FLAC) — works when files are one level down in subdirs."
    )
    parser.add_argument("directory", type=Path, help="Root folder containing the clips")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search ALL subdirectories (not just one level deep)")
    parser.add_argument("--output", "-o", type=Path, default="duplicate_pairs.txt",
                        help="Output file (default: duplicate_pairs.txt)")
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    filepaths = collect_audio_files(root, args.recursive)

    if not filepaths:
        print("No WAV or FLAC files found (checked root + immediate subfolders).")
        print("Use --recursive if your files are nested deeper.")
        return

    print(f"Found {len(filepaths)} audio files across subfolders.\n")

    pairs = find_similar_pairs(filepaths)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        for a, b, score in pairs:
            f.write(f"{score:.6f}\t{a.relative_to(root)}\t{b.relative_to(root)}\n")

    if pairs:
        print(f"\nFound {len(pairs)} near-duplicate pair(s)!")
        print(f"Results → {args.output.resolve()}\n")
        print("Top 10:")
        for a, b, s in pairs[:10]:
            print(f"  {s:.5f} | {a.relative_to(root)}  <->  {b.relative_to(root)}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs)-10} more.")
    else:
        print("\nNo duplicates found with the current strict thresholds.")
        args.output.touch()
        print(f"Empty file created → {args.output.resolve()}")


if __name__ == "__main__":
    main()