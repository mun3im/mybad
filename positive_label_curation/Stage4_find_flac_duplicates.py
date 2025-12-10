#!/usr/bin/env python3
"""
Detect near-duplicate 3-second audio clips (WAV & FLAC) across subfolders.
Moves perfect duplicates (1.000 similarity) to a quarantine folder.
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

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
PERFECT_DUPLICATE_THRESHOLD = 0.999999  # Account for floating-point precision


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
    """Finds and categorizes duplicate audio files."""

    def __init__(self, filepaths: List[Path]):
        self.filepaths = filepaths
        self.embeddings = []
        self.valid_paths = []

    def compute_embeddings(self):
        """Compute embeddings for all files."""
        print(f"\nComputing embeddings for {len(self.filepaths)} files...")

        for fp in tqdm(self.filepaths, desc="Embedding", unit="file"):
            emb = AudioEmbedder.compute(fp)
            if emb is not None:
                self.embeddings.append(emb)
                self.valid_paths.append(fp)

        print(f"Embedded {len(self.valid_paths)}/{len(self.filepaths)} files successfully.")

    def find_pairs(self) -> Tuple[List[Tuple[Path, Path, float]], List[Tuple[Path, Path]]]:
        """
        Find similar pairs and perfect duplicates.
        Returns: (similar_pairs, perfect_duplicates)
        """
        if len(self.valid_paths) < 2:
            return [], []

        similar_pairs = []
        perfect_duplicates = []

        total = len(self.valid_paths) * (len(self.valid_paths) - 1) // 2
        print("Comparing pairs...")

        with tqdm(total=total, desc="Comparing", unit="pair") as pbar:
            for i in range(len(self.valid_paths)):
                for j in range(i + 1, len(self.valid_paths)):
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

                    pbar.update(1)

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs, perfect_duplicates


class QuarantineManager:
    """Manages quarantine folder and file movements."""

    def __init__(self, root: Path):
        self.root = root
        self.quarantine_dir = root / "quarantine"

    def quarantine_duplicates(self, perfect_duplicates: List[Tuple[Path, Path]], dry_run: bool = False) -> int:
        """
        Move one file from each perfect duplicate pair to quarantine.
        Returns number of files moved (or would-be moved in dry-run).
        """
        if not perfect_duplicates:
            return 0

        if not dry_run:
            self.quarantine_dir.mkdir(exist_ok=True)
        moved_count = 0
        moved_files = set()

        print(f"\n{'[DRY RUN] Would quarantine' if dry_run else 'Quarantining'} {len(perfect_duplicates)} perfect duplicate pair(s)...")

        for file1, file2 in perfect_duplicates:
            # Determine which file to move (prefer file2, but move file1 if file2 was already moved)
            if file2 not in moved_files:
                target_file = file2
            elif file1 not in moved_files:
                target_file = file1
            else:
                # Both files already moved in previous pairs, skip
                continue

            moved_files.add(target_file)

            # Preserve subfolder structure in quarantine
            rel_path = target_file.relative_to(self.root)
            quarantine_target = self.quarantine_dir / rel_path

            if dry_run:
                print(f"  [DRY RUN] Would move: {rel_path}")
            else:
                quarantine_target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(target_file), str(quarantine_target))
                    moved_count += 1
                    print(f"  Moved: {rel_path}")
                except Exception as e:
                    print(f"  Failed to move {rel_path}: {e}", file=sys.stderr)

        if dry_run:
            return len(moved_files)
        return moved_count


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
        description="Find near-duplicate 3-second clips (WAV/FLAC) and quarantine perfect matches."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root folder containing the clips"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search ALL subdirectories (not just one level deep)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="duplicate_pairs.txt",
        help="Output file (default: duplicate_pairs.txt)"
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
    return parser.parse_args()


def main():
    """Main execution function."""
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
    finder.compute_embeddings()
    similar_pairs, perfect_duplicates = finder.find_pairs()

    # Quarantine perfect duplicates
    moved_count = 0
    if not args.no_quarantine and perfect_duplicates:
        manager = QuarantineManager(root)
        moved_count = manager.quarantine_duplicates(perfect_duplicates, dry_run=args.dry_run)

    # Write and display results
    ResultsWriter.write(args.output, root, similar_pairs, moved_count)
    ResultsWriter.print_summary(similar_pairs, moved_count, root, args.output)


if __name__ == "__main__":
    main()