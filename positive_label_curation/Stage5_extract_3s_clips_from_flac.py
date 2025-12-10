#!/usr/bin/env python3
"""
extract_clips_from_flac_updated.py

Scan an input folder tree produced by xc_convert_to_16k_flac and extract 3.0s FLAC
chunks (16 kHz, PCM_16).

Key changes from the original script you provided:
 - Writes 3s chunks as FLAC (not WAV).
 - Ensures the logged RMS is the RMS of the saved 3s clip *after* any clipping correction.
 - Reports the total number of clips written at the end.
 - Keeps dry-run behavior but still reports counts for the dry-run.

Usage example (same CLI flags as before):
  python extract_clips_from_flac_updated.py \
    --inroot /Volumes/Evo/XC-All-Malaysian-Birds_flac \
    --outroot /Volumes/Evo/XC-All-Malaysian-Birds_clips \
    --threshold 0.003 \
    --csv clips_log.csv \
    [--guarantee] [--dry-run] [--flatten]
"""

from pathlib import Path
import argparse
import os
import sys
import math
import time
import logging
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
from io import BytesIO
import pandas as pd

# ---------------- CONFIG / DEFAULTS ----------------
DEFAULT_SR = 16000  # final sampling rate (Hz)
WINDOW_SEC = 3.0    # 3-second windows
STEP_SEC = 0.1      # 100 ms step (fixed)
CLIPPING_CEILING = 0.99  # target peak after correction
SOFT_LIMIT_ALPHA = 5.0   # soft limiter shape parameter
MIN_SEPARATION_SEC = 1.5  # minimum temporal separation between chosen chunks
# ---------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("extract_clips")

# -------- helper utilities --------
def extract_xc_id_from_name(name: str) -> Optional[str]:
    """Try to extract integer Xeno-canto id from filename like 'xc123456' or 'xc123456_extra'."""
    import re
    m = re.search(r'xc(\d+)', name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def rms_of_segment(y: np.ndarray) -> float:
    """Compute RMS of audio vector y (mono)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def is_clipped(y: np.ndarray, threshold: float = 0.9999) -> bool:
    """Detect clipping by checking if any sample magnitude >= threshold."""
    return bool(np.any(np.abs(y) >= threshold))


def peak_scale_and_soft_limit(y: np.ndarray, ceiling: float = CLIPPING_CEILING, alpha: float = SOFT_LIMIT_ALPHA) -> np.ndarray:
    """
    1) Linear peak scaling so that max(abs(y)) <= ceiling
    2) Apply soft limiter y_out = (1/alpha) * tanh(alpha * y_scaled)
    Returns float32 array in [-1,1].
    """
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak == 0.0:
        return y
    # scale so peak <= ceiling (but if peak < ceiling we still might apply soft limiter lightly)
    scale = 1.0
    if peak > ceiling:
        scale = ceiling / peak
    y_scaled = y * scale
    # apply soft limiter
    y_limited = (1.0 / alpha) * np.tanh(alpha * y_scaled)
    # If the limiter introduces values >1 (shouldn't), clip
    y_limited = np.clip(y_limited, -1.0, 1.0)
    return y_limited


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """Convert multichannel audio to mono by averaging channels if needed."""
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)

# -------- selection helpers --------
def sliding_windows_rms(y: np.ndarray, sr: int, window_sec: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (starts_samples, rms_values) for sliding windows over y.
    starts_samples: numpy array of start sample indices
    rms_values: numpy array of RMS floats corresponding to each start
    """
    win_len = int(round(window_sec * sr))
    step = int(round(step_sec * sr))
    if win_len <= 0 or step <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    starts = np.arange(0, max(1, len(y) - win_len + 1), step, dtype=int)
    rms_list = []
    for s in starts:
        seg = y[s: s + win_len]
        rms_list.append(rms_of_segment(seg))
    return starts, np.array(rms_list, dtype=float)


def choose_diverse_chunks(starts: np.ndarray, rms_vals: np.ndarray, sr: int, num_chunks: int,
                          min_separation_sec: float, threshold: float) -> List[Tuple[int,float]]:
    """
    Choose up to num_chunks windows preferring diversity across time and above threshold.
    Returns list of (start_sample, rms) tuples.
    """
    chosen = []
    if starts.size == 0:
        return chosen
    # candidate indices
    candidate_idx = np.flatnonzero(rms_vals >= threshold)
    # if no candidates above threshold, return empty (caller may fallback)
    if candidate_idx.size == 0:
        return chosen
    # create a list of (idx, rms) and sort by rms desc
    cand = sorted(((int(i), float(rms_vals[i])) for i in candidate_idx), key=lambda x: x[1], reverse=True)
    chosen_starts = []
    min_sep_samples = int(round(min_separation_sec * sr))
    for idx, rms in cand:
        s = int(starts[idx])
        # check separation
        too_close = False
        for cs in chosen_starts:
            if abs(cs - s) < min_sep_samples:
                too_close = True
                break
        if not too_close:
            chosen.append((s, rms))
            chosen_starts.append(s)
            if len(chosen) >= num_chunks:
                break
    return chosen


def choose_best_chunks_any(starts: np.ndarray, rms_vals: np.ndarray, num_chunks: int, sr: int, min_sep_sec: float) -> List[Tuple[int,float]]:
    """Pick best num_chunks by RMS but enforce min separation. Allows below-threshold picks."""
    if starts.size == 0:
        return []
    idxs = np.argsort(rms_vals)[::-1]  # sorted by rms desc
    chosen = []
    chosen_starts = []
    min_sep_samples = int(round(min_sep_sec * sr))
    for i in idxs:
        s = int(starts[i])
        rms = float(rms_vals[i])
        too_close = False
        for cs in chosen_starts:
            if abs(cs - s) < min_sep_samples:
                too_close = True
                break
        if not too_close:
            chosen.append((s, rms))
            chosen_starts.append(s)
            if len(chosen) >= num_chunks:
                break
    return chosen

# -------- main processing per file --------
def process_file(path: Path, species: str, out_root: Path, sr_out: int, threshold: float, guarantee: bool,
                 csv_records: List[dict], dry_run: bool = False, flatten: bool = False) -> Tuple[List[dict], int]:
    """
    Process a single FLAC file and return updated csv_records and number of clips saved.
    """
    saved_count = 0
    # read file (librosa load uses soundfile backend by default)
    try:
        y, sr = librosa.load(str(path), sr=sr_out, mono=True)  # ensures resampled to sr_out and mono
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return csv_records, saved_count

    duration = librosa.get_duration(y=y, sr=sr)
    base_name = path.stem  # e.g. xc12345
    xc_id = extract_xc_id_from_name(base_name) or "unknown"
    # Determine expected number of chunks and offset rule (3s windows)
    if duration < 3.0:
        chunks_expected = 0
        offset_sec = 0.0
    elif duration <= 6.0:
        chunks_expected = 1
        offset_sec = 0.0
    elif duration <= 12.0:
        chunks_expected = 2
        offset_sec = 0.0
    else:
        # Long files: skip first 3s to avoid possible voice annotation
        chunks_expected = 2
        offset_sec = 3.0

    win_len_samples = int(round(WINDOW_SEC * sr))
    step_samples = int(round(STEP_SEC * sr))
    starts, rms_vals = sliding_windows_rms(y, sr, WINDOW_SEC, STEP_SEC)

    # When guarantee + flatten: ignore threshold to generate more clips for quarantine
    # Otherwise: use threshold to filter clips
    effective_threshold = 0.0 if (guarantee and flatten) else threshold

    # choose diverse chunks above effective threshold
    candidates = choose_diverse_chunks(starts, rms_vals, sr, chunks_expected, MIN_SEPARATION_SEC, effective_threshold)

    # if not enough candidates and guarantee is requested, pick best remaining (allow below threshold)
    if len(candidates) < chunks_expected and guarantee:
        fallback = choose_best_chunks_any(starts, rms_vals, chunks_expected, sr, MIN_SEPARATION_SEC)
        # merge unique starts, prioritize previously chosen ones
        chosen_dict = {s: rms for s, rms in candidates}
        for s, rms in fallback:
            if s not in chosen_dict and len(chosen_dict) < chunks_expected:
                chosen_dict[s] = rms
        candidates = sorted([(s, chosen_dict[s]) for s in chosen_dict], key=lambda x: x[0])

    # if still empty and not guarantee, we may skip
    if not candidates:
        logger.info(f"No candidate chunks >= threshold for {path.name}; skipping (set --guarantee to force one)")
        return csv_records, saved_count

    # Save chosen chunks as FLAC with start_ms suffix
    for s_samples, rms_pre in candidates:
        start_sec = s_samples / sr
        start_ms = int(round(start_sec * 1000.0))

        if flatten:
            out_fname = f"{base_name}_{start_ms}.wav"
            out_path = out_root / out_fname
        else:
            out_species_dir = out_root / species
            out_species_dir.mkdir(parents=True, exist_ok=True)
            out_fname = f"{base_name}_{start_ms}.wav"
            out_path = out_species_dir / out_fname

        if dry_run:
            logger.info(f"[DRY] Would save {out_path} (start_s={start_sec:.3f}, pre_rms={rms_pre:.6f})")
            # For dry-run, keep the precomputed rms as best estimate
            csv_records.append({
                "source_file": str(path),
                "species": species,
                "start_ms": start_ms,
                "rms": float(rms_pre),
                "out_path": str(out_path),
                "id": xc_id
            })
            saved_count += 1
            continue

        # Extract chunk and apply clipping correction if needed
        s = s_samples
        e = s + win_len_samples
        seg = y[s:e]

        clipped = is_clipped(seg, threshold=0.9999)
        if clipped:
            logger.info(f"Clipping detected in {path.name} at start {start_sec:.3f}s (pre RMS {rms_pre:.6f}) - applying correction")
            seg = peak_scale_and_soft_limit(seg, ceiling=CLIPPING_CEILING, alpha=SOFT_LIMIT_ALPHA)

        # ensure dtype float32
        seg_to_write = seg.astype(np.float32)
        # recompute RMS of the final segment (this is what we'll log)
        final_rms = rms_of_segment(seg_to_write)

        try:
            # write wav (soundfile infers format from extension too)
            sf.write(str(out_path), seg_to_write, sr, format='WAV', subtype='PCM_16')
            logger.info(f"Saved: {out_path} (start {start_sec:.3f}s, rms={final_rms:.6f}, clipped={clipped})")
            csv_records.append({
                "source_file": str(path),
                "species": species,
                "start_ms": start_ms,
                "rms": float(final_rms),
                "out_path": str(out_path),
                "id": xc_id
            })
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to write {out_path}: {e}")

    return csv_records, saved_count


def postprocess_flatten_quarantine(csv_path: Path, output_root: Path, dry_run: bool = False):
    """If flatten mode was used, enforce 25k loudest clips; rest go to quarantine."""
    logger.info("Running post-processing for --flatten: sorting by RMS and quarantining excess clips...")

    df = pd.read_csv(csv_path)
    total_clips = len(df)
    logger.info(f"Total clips generated: {total_clips}")

    if total_clips <= 25000:
        logger.info("≤25,000 clips — no quarantining needed.")
        return

    # Sort by RMS descending
    df = df.sort_values(by="rms", ascending=False).reset_index(drop=True)

    # Split into keep and quarantine
    keep_df = df.head(25000)
    quarantine_df = df.iloc[25000:]

    quarantine_dir = output_root / "quarantine"
    if not dry_run:
        quarantine_dir.mkdir(exist_ok=True)

    # Move quarantined files
    for _, row in quarantine_df.iterrows():
        src_path = Path(row["out_path"])
        dst_path = quarantine_dir / src_path.name
        if src_path.exists():
            if dry_run:
                logger.info(f"[DRY] Would move {src_path} → {dst_path}")
            else:
                try:
                    src_path.rename(dst_path)
                except Exception as e:
                    logger.error(f"Failed to move {src_path} to quarantine: {e}")
        else:
            logger.warning(f"Quarantine file not found (already moved?): {src_path}")

    # Overwrite CSV to keep only top 25k
    if not dry_run:
        keep_df.to_csv(csv_path, index=False)
        logger.info(f"CSV updated to contain only top 25,000 loudest clips (saved to {csv_path})")
    else:
        logger.info(f"[DRY] Would truncate CSV to 25,000 rows (top RMS)")

    logger.info(f"Quarantined {len(quarantine_df)} clips into {quarantine_dir}")

# -------- main loop over folders --------
def main():
    parser = argparse.ArgumentParser(description="Extract 3s WAV clips from converted FLAC files (16k) with smart selection.")
    parser.add_argument("--inroot", required=True, help="Root folder containing species subfolders with FLAC files")
    parser.add_argument("--outroot", required=True, help="Destination root for extracted FLAC clips")
    parser.add_argument("--threshold", type=float, default=0.003, help="RMS threshold to consider a 3s window non-silent (default 0.003)")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Output sampling rate, default 16000")
    parser.add_argument("--csv", default="clips_log.csv", help="CSV path to write clip metadata")
    parser.add_argument("--guarantee", action="store_true", help="Guarantee at least one saved clip per file (pick best if none exceed threshold)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; only simulate and write csv entries")
    parser.add_argument("--flatten", action="store_true",
                        help="Place all output clips directly in output-root (no species subfolders)")
    args = parser.parse_args()

    input_root = Path(args.inroot)
    output_root = Path(args.outroot)
    sr_out = int(args.sr)
    threshold = float(args.threshold)
    guarantee = bool(args.guarantee)
    csv_path = Path(args.csv)
    dry_run = bool(args.dry_run)

    if not input_root.exists():
        logger.error(f"Input root does not exist: {input_root}")
        sys.exit(2)
    output_root.mkdir(parents=True, exist_ok=True)

    # iterate species subfolders (one level)
    csv_records: List[dict] = []
    total_saved = 0
    species_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    logger.info(f"Found {len(species_dirs)} species folders under {input_root}")

    for species_dir in species_dirs:
        species_name = species_dir.name
        flac_files = sorted([p for p in species_dir.glob("*.flac")])
        logger.info(f"Processing species {species_name}: {len(flac_files)} files")
        for i, fpath in enumerate(flac_files, start=1):
            logger.debug(f"Processing file {i}/{len(flac_files)}: {fpath.name}")
            try:
                csv_records, saved = process_file(
                    fpath, species_name, output_root, sr_out, threshold, guarantee,
                    csv_records, dry_run=dry_run, flatten=args.flatten
                )
                total_saved += saved
            except Exception as e:
                logger.exception(f"Unexpected error processing {fpath}: {e}")

            # optional: flush CSV periodically to avoid data loss for long runs
            if len(csv_records) >= 1000:
                # append to disk and clear
                df_partial = pd.DataFrame(csv_records)
                if csv_path.exists():
                    df_partial.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df_partial.to_csv(csv_path, mode="w", header=True, index=False)
                logger.info(f"Flushed {len(csv_records)} records to {csv_path}")
                csv_records.clear()

    # Final write of any remaining records (as before)
    if csv_records:
        df_last = pd.DataFrame(csv_records)
        if csv_path.exists():
            df_last.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_last.to_csv(csv_path, mode="w", header=True, index=False)
        logger.info(f"Wrote final {len(csv_records)} records to {csv_path}")

    # >>> NEW: Post-processing for --flatten <<<
    if args.flatten:
        postprocess_flatten_quarantine(csv_path, output_root, dry_run=dry_run)

    logger.info(f"Processing complete. Total clips written: {total_saved}")
    print(f"Total clips written: {total_saved}")

if __name__ == "__main__":
    main()
