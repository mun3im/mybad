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
        description="Stage 7: Move excluded files to quarantine subdirectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PREREQUISITE:
  Stage6 must be run first to generate balanced_clips.csv

PURPOSE:
  Stage6 creates CSV listing clips to keep (metadata balancing only).
  This script physically moves excluded files to quarantine/ subdirectory.

EXAMPLES:
  # Move excluded files to quarantine after Stage6
  python Stage7_move_to_quarantine.py --input-csv balanced_clips.csv --outroot clips/

  # Preview what would be moved (dry-run)
  python Stage7_move_to_quarantine.py --input-csv balanced_clips.csv --outroot clips/ --dry-run

NOTE:
  This is safer than deleting - excluded clips are preserved in quarantine/
  for potential future use.
        """
    )

    # Required arguments
    parser.add_argument("--input-csv", required=True, metavar="FILE",
                        help="Balanced clips CSV from Stage6 (e.g., balanced_clips.csv)")
    parser.add_argument("--outroot", required=True, metavar="DIR",
                        help="Root directory containing audio clips")

    # Processing options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be moved without actually moving files")
    args = parser.parse_args()

    # Paths
    balanced_csv = Path(args.input_csv)
    outroot = Path(args.outroot)
    quarantine_dir = outroot / "quarantine"
    dry_run = args.dry_run

    if not balanced_csv.exists():
        print(f"ERROR: Balanced CSV not found: {balanced_csv}")
        print("Please run Stage6 first to generate balanced_clips.csv")
        sys.exit(1)

    if not outroot.exists():
        print(f"ERROR: Output directory not found: {outroot}")
        sys.exit(1)

    # Create quarantine directory if it doesn't exist
    if not dry_run:
        quarantine_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stage 7: Move Excluded Files to Quarantine")
    print(f"{'='*60}")
    print(f"Balanced CSV: {balanced_csv}")
    print(f"Output directory: {outroot}")
    print(f"Quarantine directory: {quarantine_dir}")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be moved)")
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
        if dry_run:
            print(f"\n[DRY RUN] Would move {len(files_to_move):,} files to quarantine")
            print("\nSample files that would be moved (first 5):")
            for file_path in files_to_move[:5]:
                print(f"  {file_path.name}")
            if len(files_to_move) > 5:
                print(f"  ... and {len(files_to_move) - 5:,} more")
        else:
            print("\nMoving files to quarantine...")
            for file_path in tqdm(files_to_move, desc="Moving files", unit="file"):
                dest_path = quarantine_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
            print(f"\nSuccessfully moved {len(files_to_move):,} files to quarantine")
    else:
        print("\nNo files to move!")

    # Verify
    if not dry_run:
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
    else:
        print(f"\n[DRY RUN] Verification skipped")
        print(f"[DRY RUN] After moving, {len(keep_files):,} files would remain in destination")

if __name__ == "__main__":
    main()
