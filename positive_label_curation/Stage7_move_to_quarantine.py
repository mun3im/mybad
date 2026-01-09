#!/usr/bin/env python3
"""
Stage7_move_to_quarantine.py

Creates final dataset directory with balanced clips from Stage6.

Input: Stage6_balanced_clips.csv (default)
Output: Creates new dataset directory (e.g., dataset_20000_001)

Purpose:
 - Stage6 creates a CSV listing which clips to keep (metadata balancing only)
 - This script physically copies/moves the selected files to a new dataset directory
 - Auto-increments directory name if it already exists (dataset_20000_001, dataset_20000_002, etc.)
 - Keeps only the balanced subset in the dataset directory

Key features:
 - Reads Stage6_balanced_clips.csv to identify files to keep
 - Creates new dataset directory with target size in name
 - Auto-increments suffix if directory exists
 - Copies selected files to dataset directory
 - Progress bar for large operations
 - Verifies final count matches expected

Usage example:
  # After Stage6, run Stage7 to create final dataset
  python Stage7_move_to_quarantine.py --outroot /path/to/clips

  # Custom dataset directory name
  python Stage7_move_to_quarantine.py --outroot /path/to/clips --dataset-dir my_dataset

Note: Original clips remain in outroot, balanced subset is copied to dataset directory.
"""
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import sys

def find_next_dataset_dir(base_dir: Path, target_size: int) -> Path:
    """
    Find next available dataset directory name with auto-increment.
    Format: dataset_{target_size}_{counter:03d}
    """
    counter = 1
    while True:
        dataset_dir = base_dir / f"dataset_{target_size}_{counter:03d}"
        if not dataset_dir.exists():
            return dataset_dir
        counter += 1


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Create final dataset directory with balanced clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PREREQUISITE:
  Stage6 must be run first to generate Stage6_balanced_clips.csv

PURPOSE:
  Creates final dataset directory with only the balanced clips from Stage6.
  Auto-increments directory name if it already exists.

EXAMPLES:
  # Create dataset directory (uses defaults)
  python Stage7_move_to_quarantine.py --outroot clips/

  # Custom input CSV and dataset name
  python Stage7_move_to_quarantine.py --input-csv Stage6_balanced_clips.csv \\
    --outroot clips/ --dataset-dir my_custom_dataset

  # Preview what would be copied (dry-run)
  python Stage7_move_to_quarantine.py --outroot clips/ --dry-run

NOTE:
  Original clips remain in outroot. Balanced subset is copied to dataset directory.
        """
    )

    # Arguments
    parser.add_argument("--input-csv", default="Stage6_balanced_clips.csv", metavar="FILE",
                        help="Balanced clips CSV from Stage6 (default: Stage6_balanced_clips.csv)")
    parser.add_argument("--outroot", required=True, metavar="DIR",
                        help="Root directory containing audio clips")
    parser.add_argument("--dataset-dir", default=None, metavar="DIR",
                        help="Dataset directory name (default: auto-generated as dataset_{size}_{counter})")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them (default: copy)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually copying/moving files")
    args = parser.parse_args()

    # Paths
    balanced_csv = Path(args.input_csv)
    outroot = Path(args.outroot)
    dry_run = args.dry_run

    if not balanced_csv.exists():
        print(f"ERROR: Balanced CSV not found: {balanced_csv}")
        print("Please run Stage6 first to generate Stage6_balanced_clips.csv")
        sys.exit(1)

    if not outroot.exists():
        print(f"ERROR: Output directory not found: {outroot}")
        sys.exit(1)

    print("Loading balanced clips list...")
    df_balanced = pd.read_csv(balanced_csv)

    # Get set of filenames to copy (clip_filename column from Stage5)
    if 'clip_filename' in df_balanced.columns:
        keep_files = set(df_balanced['clip_filename'].values)
    elif 'out_filename' in df_balanced.columns:
        keep_files = set(df_balanced['out_filename'].values)
    else:
        print("ERROR: CSV must contain either 'clip_filename' or 'out_filename' column")
        sys.exit(1)

    target_size = len(keep_files)
    print(f"Target dataset size: {target_size:,} files")

    # Determine dataset directory
    if args.dataset_dir:
        dataset_dir = outroot.parent / args.dataset_dir
    else:
        dataset_dir = find_next_dataset_dir(outroot.parent, target_size)

    print(f"\n{'='*60}")
    print(f"Stage 7: Create Final Dataset Directory")
    print(f"{'='*60}")
    print(f"Balanced CSV: {balanced_csv}")
    print(f"Source directory: {outroot}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Files to copy: {target_size:,}")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be copied)")
    print(f"{'='*60}\n")

    # Create dataset directory
    if not dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created dataset directory: {dataset_dir}")

    # Copy files to dataset directory
    print("\nCopying files to dataset directory...")
    copied_count = 0
    missing_files = []

    files_to_copy = []
    for filename in keep_files:
        source_file = outroot / filename
        if source_file.exists():
            files_to_copy.append((source_file, filename))
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files):,} files not found in source directory")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"  Missing: {f}")
        else:
            for f in missing_files[:5]:
                print(f"  Missing: {f}")
            print(f"  ... and {len(missing_files) - 5:,} more")

    if dry_run:
        print(f"\n[DRY RUN] Would copy {len(files_to_copy):,} files to {dataset_dir}")
        print("\nSample files that would be copied (first 5):")
        for source_file, filename in files_to_copy[:5]:
            print(f"  {filename}")
        if len(files_to_copy) > 5:
            print(f"  ... and {len(files_to_copy) - 5:,} more")
    else:
        for source_file, filename in tqdm(files_to_copy, desc="Copying files", unit="file"):
            dest_file = dataset_dir / filename
            shutil.copy2(str(source_file), str(dest_file))
            copied_count += 1

        print(f"\n✓ Successfully copied {copied_count:,} files to dataset directory")

        # Save a copy of the balanced CSV in the dataset directory
        dest_csv = dataset_dir / "dataset_manifest.csv"
        df_balanced.to_csv(dest_csv, index=False)
        print(f"✓ Saved dataset manifest: {dest_csv}")

    # Verify
    if not dry_run:
        print("\nVerifying...")
        dataset_files = list(dataset_dir.glob('*.wav')) + list(dataset_dir.glob('*.flac'))
        print(f"Files in dataset directory: {len(dataset_files):,}")
        print(f"Expected: {target_size:,}")

        if len(dataset_files) == target_size:
            print(f"\n✓ Success! Dataset ready at: {dataset_dir}")
        else:
            print(f"\n⚠ Warning: File count mismatch! Expected {target_size:,}, got {len(dataset_files):,}")
    else:
        print(f"\n[DRY RUN] Would create dataset at: {dataset_dir}")
        print(f"[DRY RUN] Final dataset would contain {len(files_to_copy):,} files")

if __name__ == "__main__":
    main()
