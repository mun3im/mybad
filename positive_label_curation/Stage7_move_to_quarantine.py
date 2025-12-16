#!/usr/bin/env python3
"""
Stage7_move_to_quarantine.py

Move audio files excluded from balanced dataset to quarantine subdirectory.

PREREQUISITE:
 - Stage6 must be run first to generate balanced_clips.csv

Purpose:
 - Stage6 creates a CSV listing which clips to keep (metadata balancing only)
 - This script physically moves the excluded files to quarantine/
 - Keeps only the balanced subset in the main directory

Key features:
 - Reads balanced_clips.csv to identify files to keep
 - Moves all other files to quarantine/ subdirectory
 - Verifies file counts match expected values
 - Progress bar for large operations

Usage example:
  # First, run Stage6 to create balanced_clips.csv
  python Stage6_balance_species.py \
    --csv clips_log.csv \
    --outroot /path/to/clips \
    --target-size 25000

  # Then run Stage7 to move excluded files
  python Stage7_move_to_quarantine.py \
    --csv balanced_clips.csv \
    --outroot /path/to/clips

Note: This script is safer than deleting files - excluded clips are preserved
in quarantine/ for potential future use.
"""
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Move files excluded from balanced dataset to quarantine subdirectory"
    )
    parser.add_argument("--csv", required=True, help="Balanced clips CSV from Stage6 (e.g., balanced_clips.csv)")
    parser.add_argument("--outroot", required=True, help="Root directory containing audio clips")
    args = parser.parse_args()

    # Paths
    balanced_csv = Path(args.csv)
    outroot = Path(args.outroot)
    quarantine_dir = outroot / "quarantine"

    if not balanced_csv.exists():
        print(f"ERROR: Balanced CSV not found: {balanced_csv}")
        print("Please run Stage6 first to generate balanced_clips.csv")
        sys.exit(1)

    if not outroot.exists():
        print(f"ERROR: Output directory not found: {outroot}")
        sys.exit(1)

    # Create quarantine directory if it doesn't exist
    quarantine_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 7: Move Excluded Files to Quarantine")
    print(f"{'='*60}")
    print(f"Balanced CSV: {balanced_csv}")
    print(f"Output directory: {outroot}")
    print(f"Quarantine directory: {quarantine_dir}")
    print(f"{'='*60}\n")

    print("Loading balanced clips list...")
    df_balanced = pd.read_csv(balanced_csv)

    # Get set of filenames to KEEP (just the filename, not full path)
    keep_files = set(df_balanced['out_filename'].values)
    print(f"Files to keep: {len(keep_files):,}")

    # Get all audio files in destination directory (excluding quarantine subdir)
    print("\nScanning destination directory...")
    all_files = []
    for ext in ['*.flac', '*.wav']:
        all_files.extend(outroot.glob(ext))

    # Filter out files already in quarantine
    all_files = [f for f in all_files if f.parent == outroot]
    print(f"Total files in destination: {len(all_files):,}")

    # Identify files to move
    files_to_move = []
    for file_path in all_files:
        if file_path.name not in keep_files:
            files_to_move.append(file_path)

    print(f"Files to move to quarantine: {len(files_to_move):,}")
    print(f"Files to keep in destination: {len(all_files) - len(files_to_move):,}")

    # Move files to quarantine
    if files_to_move:
        print("\nMoving files to quarantine...")
        for file_path in tqdm(files_to_move, desc="Moving files", unit="file"):
            dest_path = quarantine_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
        print(f"\nSuccessfully moved {len(files_to_move):,} files to quarantine")
    else:
        print("\nNo files to move!")

    # Verify
    print("\nVerifying...")
    remaining_files = list(outroot.glob('*.flac')) + list(outroot.glob('*.wav'))
    remaining_files = [f for f in remaining_files if f.parent == outroot]
    quarantined_files = list(quarantine_dir.glob('*.flac')) + list(quarantine_dir.glob('*.wav'))

    print(f"Files remaining in destination: {len(remaining_files):,}")
    print(f"Files in quarantine: {len(quarantined_files):,}")
    print(f"Expected in destination: {len(keep_files):,}")

    if len(remaining_files) == len(keep_files):
        print("\n✓ Success! File counts match.")
    else:
        print(f"\n⚠ Warning: File count mismatch! Expected {len(keep_files):,}, got {len(remaining_files):,}")

if __name__ == "__main__":
    main()
