#!/usr/bin/env python3
"""
Detect near-duplicate 3-second audio clips (WAV & FLAC) across subfolders.
Moves perfect duplicates (1.000 similarity) to a quarantine folder.
"""

import argparse
import platform
import sys
import shutil
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Set
import time
import faiss

import numpy as np
import librosa
from tqdm import tqdm

# ------------------ CONFIG ------------------
TARGET_SR = 16000
TARGET_DURATION = 3.0
N_MELS = 128
HOP_LENGTH = 128
MIN_DURATION = 3.0

MEAN_SIM_THRESHOLD = 0.997
MIN_SIM_THRESHOLD = 0.985
P5_SIM_THRESHOLD = 0.992
PERFECT_DUPLICATE_THRESHOLD = 0.999


# --------------------------------------------


class AudioEmbedder:
    """Handles audio file loading and embedding computation."""

    @staticmethod
    def compute(filepath: Path) -> Optional[np.ndarray]:
        """Compute normalized mel-spectrogram embedding for an audio file."""
        try:
            y, _ = librosa.load(
                filepath, sr=TARGET_SR, mono=True, res_type='kaiser_fast'
            )
        except Exception as e:
            print(f"Failed loading {filepath.name}: {e}", file=sys.stderr)
            return None

        if len(y) / TARGET_SR < MIN_DURATION:
            return None

        # Force exactly 3.00 seconds
        target_samples = int(TARGET_DURATION * TARGET_SR)
        y = AudioEmbedder._normalize_length(y, target_samples)

        # Compute mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_mels=N_MELS,
            hop_length=HOP_LENGTH, n_fft=512, fmax=8000
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-frame L2 normalization for cosine similarity
        return AudioEmbedder._normalize_frames(log_mel)

    @staticmethod
    def _normalize_length(y: np.ndarray, target_samples: int) -> np.ndarray:
        """Pad or trim audio to exact target length."""
        if len(y) > target_samples:
            return y[:target_samples]
        return np.pad(y, (0, target_samples - len(y)), mode='constant')

    @staticmethod
    def _normalize_frames(log_mel: np.ndarray) -> np.ndarray:
        """L2 normalize each frame for cosine similarity."""
        norms = np.linalg.norm(log_mel, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return (log_mel / norms).astype(np.float32)


class SimilarityCalculator:
    """Calculates frame-wise cosine similarity between embeddings."""

    @staticmethod
    def compute_metrics(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float, float]:
        """Returns (mean_sim, min_sim, p5_sim) between two embeddings."""
        min_frames = min(emb1.shape[1], emb2.shape[1])
        if min_frames == 0:
            return 0.0, 0.0, 0.0

        sims = np.clip(
            np.sum(emb1[:, :min_frames] * emb2[:, :min_frames], axis=0),
            -1.0, 1.0
        )
        return (
            float(np.mean(sims)),
            float(np.min(sims)),
            float(np.percentile(sims, 5))
        )

    @staticmethod
    def is_similar(mean_sim: float, min_sim: float, p5_sim: float) -> bool:
        """Check if similarity metrics exceed thresholds."""
        return (
                mean_sim >= MEAN_SIM_THRESHOLD and
                min_sim >= MIN_SIM_THRESHOLD and
                p5_sim >= P5_SIM_THRESHOLD
        )


class DuplicateFinder:
    """Finds and categorizes duplicate audio files using FAISS."""

    def __init__(self, filepaths: List[Path], top_k: int = 6):
        self.filepaths = filepaths
        self.embeddings = []       # framewise (128 x T)
        self.clip_embeddings = []  # clip-level (D,)
        self.valid_paths = []
        self.top_k = top_k

    def compute_embeddings(self):
        print(f"\nComputing embeddings for {len(self.filepaths)} files...")

        for fp in tqdm(self.filepaths, desc="Embedding", unit="file"):
            emb = AudioEmbedder.compute(fp)
            if emb is not None:
                self.embeddings.append(emb)
                self.clip_embeddings.append(self._to_clip_embedding(emb))
                self.valid_paths.append(fp)

        self.clip_embeddings = np.vstack(self.clip_embeddings).astype("float32")
        faiss.normalize_L2(self.clip_embeddings)

        print(f"Embedded {len(self.valid_paths)}/{len(self.filepaths)} files successfully.")

    @staticmethod
    def _to_clip_embedding(emb: np.ndarray) -> np.ndarray:
        """
        Convert framewise mel embedding (128 x T)
        to clip-level descriptor using mean+std pooling.
        """
        mu = np.mean(emb, axis=1)
        std = np.std(emb, axis=1)
        return np.concatenate([mu, std])

    def find_pairs(self) -> Tuple[List[Tuple[Path, Path, float]], List[Tuple[Path, Path]]]:
        if len(self.valid_paths) < 2:
            return [], []

        N, D = self.clip_embeddings.shape
        print(f"\nBuilding FAISS index (N={N}, D={D})...")

        index = faiss.IndexFlatIP(D)  # cosine after L2 norm
        index.add(self.clip_embeddings)

        print("Searching nearest neighbors...")
        S, I = index.search(self.clip_embeddings, self.top_k)

        similar_pairs = []
        perfect_duplicates = []
        visited = set()

        print("Verifying candidate pairs...")

        for i in tqdm(range(N), desc="Verifying", unit="file"):
            for r in range(1, self.top_k):  # skip self
                j = int(I[i, r])
                sim = float(S[i, r])

                if j <= i:
                    continue
                if (i, j) in visited:
                    continue
                visited.add((i, j))

                # quick rejection
                if sim < MIN_SIM_THRESHOLD:
                    continue

                # ---- Upgrade (B): instant perfect-duplicate check ----
                if np.allclose(self.clip_embeddings[i], self.clip_embeddings[j], atol=1e-7):
                    perfect_duplicates.append(
                        (self.valid_paths[i], self.valid_paths[j])
                    )
                    continue
                # ------------------------------------------------------

                mean_sim, min_sim, p5_sim = SimilarityCalculator.compute_metrics(
                    self.embeddings[i], self.embeddings[j]
                )

                if mean_sim >= PERFECT_DUPLICATE_THRESHOLD:
                    perfect_duplicates.append(
                        (self.valid_paths[i], self.valid_paths[j])
                    )

                elif SimilarityCalculator.is_similar(mean_sim, min_sim, p5_sim):
                    similar_pairs.append(
                        (self.valid_paths[i], self.valid_paths[j], mean_sim)
                    )

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs, perfect_duplicates


class QuarantineManager:
    """Manages quarantine folder and file movements."""

    def __init__(self, root: Path):
        self.root = root
        self.quarantine_dir = root / "quarantine"

    @staticmethod
    def extract_xc_number(filepath: Path) -> int:
        """
        Extract XC number from filename (e.g., 'xc123456_A.flac' -> 123456).
        Returns 0 if no XC number found.
        """
        try:
            name = filepath.stem.lower()
            if name.startswith('xc'):
                # Extract digits after 'xc' and before '_'
                parts = name[2:].split('_')
                if parts:
                    return int(parts[0])
        except (ValueError, IndexError):
            pass
        return 0

    def quarantine_duplicates(self, perfect_duplicates: List[Tuple[Path, Path]], dry_run: bool = False) -> Tuple[int, Set[int]]:
        """
        Move one file from each perfect duplicate pair to quarantine.
        Returns (number of files moved, set of quarantined XC numbers).
        """
        if not perfect_duplicates:
            return 0, set()

        if not dry_run:
            self.quarantine_dir.mkdir(exist_ok=True)
        moved_count = 0
        moved_files = set()
        quarantined_xc_numbers = set()

        print(f"\n{'[DRY RUN] Would quarantine' if dry_run else 'Quarantining'} {len(perfect_duplicates)} perfect duplicate pair(s)...")

        for file1, file2 in perfect_duplicates:
            # Quarantine the NEWER file (higher XC number), keep the older one
            xc1 = self.extract_xc_number(file1)
            xc2 = self.extract_xc_number(file2)

            # Determine which file to quarantine based on XC number
            if xc1 > xc2:
                target_file = file1  # file1 is newer, quarantine it
            elif xc2 > xc1:
                target_file = file2  # file2 is newer, quarantine it
            else:
                # Same or no XC numbers, use original logic (prefer quarantining file2)
                target_file = file2 if file2 not in moved_files else file1

            # Skip if this file was already quarantined in a previous pair
            if target_file in moved_files:
                continue

            moved_files.add(target_file)

            # Track XC number of quarantined file
            xc_num = self.extract_xc_number(target_file)
            if xc_num > 0:
                quarantined_xc_numbers.add(xc_num)

            # Preserve subfolder structure in quarantine
            rel_path = target_file.relative_to(self.root)
            quarantine_target = self.quarantine_dir / rel_path

            if dry_run:
                print(f"  [DRY RUN] Would move: {rel_path} (XC{xc_num})")
            else:
                quarantine_target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(target_file), str(quarantine_target))
                    moved_count += 1
                    print(f"  Moved: {rel_path} (XC{xc_num})")
                except Exception as e:
                    print(f"  Failed to move {rel_path}: {e}", file=sys.stderr)

        if dry_run:
            return len(moved_files), quarantined_xc_numbers
        return moved_count, quarantined_xc_numbers


class FileCollector:
    """Collects audio files from directory structure."""

    @staticmethod
    def collect(root: Path, recursive: bool) -> List[Path]:
        """
        Collect all .wav and .flac files.
        Default: root + immediate subfolders only
        --recursive: full recursion
        """
        pattern = "**/*" if recursive else "*"
        candidates = FileCollector._find_files(root, pattern)

        # Check one level deeper if nothing found and not recursive
        if not candidates and not recursive:
            candidates = FileCollector._check_subdirs(root)

        return sorted(candidates)

    @staticmethod
    def _find_files(root: Path, pattern: str) -> List[Path]:
        """Find audio files matching pattern."""
        return [
            p for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
        ]

    @staticmethod
    def _check_subdirs(root: Path) -> List[Path]:
        """Check immediate subdirectories for audio files."""
        candidates = []
        for subdir in root.iterdir():
            if subdir.is_dir():
                candidates.extend(FileCollector._find_files(subdir, "*"))
        return candidates


class UniqueFilesWriter:
    """Creates CSV of unique (non-quarantined) files based on Stage2 CSV."""

    @staticmethod
    def create_unique_csv(
            stage2_csv: Path,
            quarantined_xc_numbers: Set[int],
            output_csv: Path
    ) -> Tuple[int, int]:
        """
        Create Stage4_unique_flacs.csv from Stage2_xc_successful_downloads.csv,
        excluding quarantined files.
        Returns (total_rows, kept_rows).
        """
        if not stage2_csv.exists():
            print(f"Warning: {stage2_csv} not found. Skipping unique CSV creation.")
            return 0, 0

        kept_rows = []
        total_rows = 0

        with open(stage2_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                total_rows += 1
                try:
                    xc_id = int(row.get('id', '0'))
                except (ValueError, TypeError):
                    # Keep rows with invalid IDs (shouldn't happen but be safe)
                    kept_rows.append(row)
                    continue

                # Keep row if XC number is NOT in quarantined set
                if xc_id not in quarantined_xc_numbers:
                    kept_rows.append(row)

        # Write unique files CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

        print(f"\n✓ Created {output_csv}")
        print(f"  Total files in Stage2: {total_rows}")
        print(f"  Quarantined: {total_rows - len(kept_rows)}")
        print(f"  Kept (unique): {len(kept_rows)}")

        return total_rows, len(kept_rows)


class ResultsWriter:
    """Writes results to output file."""

    @staticmethod
    def write(
            output_path: Path,
            root: Path,
            similar_pairs: List[Tuple[Path, Path, float]],
            perfect_count: int
    ):
        """Write similar pairs to output file."""
        with open(output_path, "w", encoding="utf-8") as f:
            if perfect_count > 0:
                f.write(f"# {perfect_count} perfect duplicate pair(s) moved to quarantine/\n\n")

            for file1, file2, score in similar_pairs:
                f.write(
                    f"{score:.6f}\t"
                    f"{file1.relative_to(root)}\t"
                    f"{file2.relative_to(root)}\n"
                )

    @staticmethod
    def print_summary(
            similar_pairs: List[Tuple[Path, Path, float]],
            perfect_count: int,
            root: Path,
            output_path: Path
    ):
        """Print summary of results."""
        if perfect_count > 0:
            print(f"\n✓ Moved {perfect_count} perfect duplicate(s) to quarantine/")

        if similar_pairs:
            print(f"\nFound {len(similar_pairs)} near-duplicate pair(s)!")
            print(f"Results → {output_path.resolve()}\n")
            print("Top 10:")
            for file1, file2, score in similar_pairs[:10]:
                print(
                    f"  {score:.5f} | {file1.relative_to(root)}  "
                    f"<->  {file2.relative_to(root)}"
                )
            if len(similar_pairs) > 10:
                print(f"  ... and {len(similar_pairs) - 10} more.")
        else:
            print("\nNo near-duplicates found with current thresholds.")
            output_path.touch()
            print(f"Empty file created → {output_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Find near-duplicate FLAC files and quarantine newer duplicates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Find duplicates and quarantine newer ones, create unique files CSV
  python Stage4_find_flac_duplicates.py /path/to/flacs \\
    --stage2-csv Stage2_xc_successful_downloads.csv \\
    --stage4-csv Stage4_unique_flacs.csv

  # Dry run to see what would be quarantined
  python Stage4_find_flac_duplicates.py /path/to/flacs \\
    --stage2-csv Stage2_xc_successful_downloads.csv --dry-run

  # Search recursively in all subdirectories
  python Stage4_find_flac_duplicates.py /path/to/flacs --recursive \\
    --stage2-csv Stage2_xc_successful_downloads.csv
        """
    )

    # Required arguments
    parser.add_argument(
        "directory",
        type=Path,
        metavar="DIR",
        help="Root folder containing the FLAC files"
    )

    # Processing options
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search ALL subdirectories (not just one level deep)"
    )
    parser.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Skip moving perfect duplicates to quarantine folder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate quarantining without actually moving files"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="duplicate_pairs.txt",
        metavar="FILE",
        help="Output file for duplicate pairs list (default: duplicate_pairs.txt)"
    )
    parser.add_argument(
        "--stage2-csv",
        type=Path,
        metavar="FILE",
        help="Path to Stage2_xc_successful_downloads.csv (for creating Stage4_unique_flacs.csv)"
    )
    parser.add_argument(
        "--stage4-csv",
        type=Path,
        default="Stage4_unique_flacs.csv",
        metavar="FILE",
        help="Output path for Stage4_unique_flacs.csv (default: Stage4_unique_flacs.csv)"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    script_start = time.perf_counter()
    args = parse_arguments()

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect files
    filepaths = FileCollector.collect(root, args.recursive)
    if not filepaths:
        print("No WAV or FLAC files found (checked root + immediate subfolders).")
        print("Use --recursive if your files are nested deeper.")
        return

    print(f"Found {len(filepaths)} audio files across subfolders.\n")

    # Find duplicates
    finder = DuplicateFinder(filepaths)
    compute_embeddings_start = time.perf_counter()
    finder.compute_embeddings()
    find_pairs_start = time.perf_counter()
    similar_pairs, perfect_duplicates = finder.find_pairs()
    find_pairs_end = time.perf_counter()

    # Quarantine perfect duplicates
    moved_count = 0
    quarantined_xc_numbers = set()
    if not args.no_quarantine and perfect_duplicates:
        manager = QuarantineManager(root)
        moved_count, quarantined_xc_numbers = manager.quarantine_duplicates(
            perfect_duplicates, dry_run=args.dry_run
        )

    # Create Stage4 unique files CSV if Stage2 CSV provided
    if args.stage2_csv:
        UniqueFilesWriter.create_unique_csv(
            args.stage2_csv,
            quarantined_xc_numbers,
            args.stage4_csv
        )

    # Write and display results
    ResultsWriter.write(args.output, root, similar_pairs, moved_count)
    ResultsWriter.print_summary(similar_pairs, moved_count, root, args.output)

    script_end = time.perf_counter()

    print(f"Platform: {platform.platform()}\n")
    print(f"Time to compute embeddings: {find_pairs_start - compute_embeddings_start:.3f}s\n")
    print(f"Time to find pairs: {find_pairs_end - find_pairs_start:.3f}s\n")
    print(f"Total script time: {script_end - script_start:.3f}s\n")

if __name__ == "__main__":
    main()