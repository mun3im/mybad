#!/usr/bin/env python3
"""
Stage3_convert_mp3_to_16k_flac.py

Converts downloaded MP3 files to mono 16kHz 16-bit FLAC format.

Input: Stage2_xc_successful_downloads.csv (default)
Outputs:
  - Stage3_xc_successful_conversion.csv (successful conversions)
  - Stage3_failed_conversion.csv (failed conversions)

Features:
- Clipping detection and correction
- Optional peak normalization
- Minimum sample rate filtering (default: 16kHz)
- Parallel processing
- Ignores non-sound files

Usage:
    python Stage3_convert_mp3_to_16k_flac.py --inroot ./data --outroot ./flac
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

# Configuration constants
TARGET_SR = 16000
DEFAULT_MIN_SR = 16000     # Only process files ≥ 16kHz
TARGET_CHANNELS = 1
TARGET_FORMAT = "flac"
TARGET_SUBTYPE = "PCM_16"
INPUT_CSV = "Stage2_xc_successful_downloads.csv"
SUCCESS_CSV = "Stage3_xc_successful_conversion.csv"
FAILED_CSV = "Stage3_failed_conversion.csv"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("audio_converter")


@dataclass
class ConversionResult:
    """Results from converting a single audio file."""
    input_path: str
    output_path: Optional[str] = None
    detected_peak: Optional[float] = None
    gain_applied: Optional[float] = None
    clipped_flag: bool = False
    status: str = "error"
    error: Optional[str] = None
    # CSV record fields
    csv_record: Optional[dict] = None


class AudioProbe:
    """Utilities for probing audio file properties."""

    @staticmethod
    def get_sample_rate(filepath: str) -> tuple[Optional[int], Optional[str]]:
        """Get native sample rate using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=nw=1",
            filepath
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            line = result.stdout.strip()

            if line.startswith("sample_rate="):
                sr = int(line.split("=", 1)[1])
                return sr, None
            return None, "Missing sample_rate in ffprobe output"

        except subprocess.CalledProcessError as e:
            return None, f"ffprobe failed: {e}"
        except FileNotFoundError:
            return None, "ffprobe not found (install ffmpeg)"
        except Exception as e:
            return None, f"Unexpected error: {e}"


class AudioProcessor:
    """Core audio processing operations using ffmpeg."""

    @staticmethod
    def decode_to_wav(in_path: str, out_wav: str, to_mono: bool = False) -> tuple[bool, Optional[str]]:
        """Decode audio file to PCM WAV."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-vn", "-f", "wav"
        ]

        if to_mono:
            cmd += ["-ac", "1"]
        cmd.append(out_wav)

        try:
            subprocess.run(cmd, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, f"Decode error: {e}"
        except FileNotFoundError:
            return False, "ffmpeg not found"

    @staticmethod
    def encode_to_flac(
            in_wav: str,
            out_path: str,
            sr: int = TARGET_SR,
            channels: int = TARGET_CHANNELS
    ) -> tuple[bool, Optional[str]]:
        """Encode WAV to FLAC with specified sample rate and channels."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_wav,
            "-ac", str(channels),
            "-ar", str(sr),
            "-compression_level", "5",
            out_path
        ]

        try:
            subprocess.run(cmd, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, f"Encode error: {e}"
        except FileNotFoundError:
            return False, "ffmpeg not found"

    @staticmethod
    def apply_gain(in_path: str, out_wav: str, gain: float) -> tuple[bool, Optional[str]]:
        """Apply linear gain using ffmpeg volume filter."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-af", f"volume={gain}",
            "-f", "wav",
            out_wav
        ]

        try:
            subprocess.run(cmd, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, f"Gain application error: {e}"
        except FileNotFoundError:
            return False, "ffmpeg not found"

    @staticmethod
    def measure_peak(wav_path: str) -> tuple[Optional[float], Optional[str]]:
        """Measure peak amplitude from WAV file."""
        try:
            data, _ = sf.read(wav_path, always_2d=False, dtype="float32")
            if data is None:
                return None, "soundfile returned None"

            peak = float(np.max(np.abs(data)))
            return peak, None

        except Exception as e:
            return None, f"Peak measurement error: {e}"


class FileConverter:
    """Handles conversion of individual audio files."""

    def __init__(
            self,
            out_root: str,
            tmp_dir: str,
            normalize_db: Optional[float] = None,
            allow_boost: bool = False,
            min_sr: Optional[int] = None
    ):
        self.out_root = out_root
        self.tmp_dir = tmp_dir
        self.normalize_db = normalize_db
        self.allow_boost = allow_boost
        self.min_sr = min_sr
        self.probe = AudioProbe()
        self.processor = AudioProcessor()

    def _generate_temp_path(self, prefix: str) -> str:
        """Generate unique temporary file path."""
        timestamp = int(time.time() * 1000)
        pid = os.getpid()
        return os.path.join(self.tmp_dir, f"{prefix}_{timestamp}_{pid}.wav")

    def _check_sample_rate(self, in_path: str) -> Optional[str]:
        """Check if file meets minimum sample rate requirement."""
        if self.min_sr is None:
            return None

        sr_native, err = self.probe.get_sample_rate(in_path)
        if err:
            return f"SR probe failed: {err}"

        if sr_native < self.min_sr:
            return f"native_sr={sr_native} < min_sr={self.min_sr}"

        return None

    def _handle_clipping(self, wav_path: str, peak: float) -> tuple[str, float, float]:
        """Apply attenuation if clipping detected."""
        if peak < 1.0:
            return wav_path, peak, 1.0

        target_peak = 0.99
        attenuation = target_peak / peak

        tmp_wav = self._generate_temp_path("gain")
        ok, err = self.processor.apply_gain(wav_path, tmp_wav, attenuation)

        if not ok:
            raise RuntimeError(f"Gain application failed: {err}")

        new_peak, err = self.processor.measure_peak(tmp_wav)
        if err:
            raise RuntimeError(f"Peak measurement after gain failed: {err}")

        # Clean up original
        self._safe_remove(wav_path)

        return tmp_wav, new_peak, attenuation

    def _apply_normalization(self, wav_path: str, current_peak: float, current_gain: float) -> tuple[str, float, float]:
        """Apply peak normalization to target dBFS."""
        if self.normalize_db is None or current_peak <= 0:
            return wav_path, current_peak, current_gain

        target_linear = 10 ** (self.normalize_db / 20.0)
        gain_needed = target_linear / current_peak

        if not self.allow_boost and gain_needed > 1.0:
            gain_needed = 1.0

        if gain_needed == 1.0:
            return wav_path, current_peak, current_gain

        tmp_wav = self._generate_temp_path("norm")
        ok, err = self.processor.apply_gain(wav_path, tmp_wav, gain_needed)

        if not ok:
            raise RuntimeError(f"Normalization failed: {err}")

        new_peak, err = self.processor.measure_peak(tmp_wav)
        if err:
            raise RuntimeError(f"Peak measurement after normalization failed: {err}")

        # Clean up previous temp file
        self._safe_remove(wav_path)

        cumulative_gain = current_gain * gain_needed
        return tmp_wav, new_peak, cumulative_gain

    def _safe_remove(self, path: str):
        """Safely remove a file, ignoring errors."""
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def convert(self, in_path: str, rel_subdir: str, csv_record: Optional[dict] = None) -> ConversionResult:
        """Convert a single audio file through the complete pipeline."""
        result = ConversionResult(input_path=in_path, csv_record=csv_record)

        try:
            # Check sample rate requirement
            sr_error = self._check_sample_rate(in_path)
            if sr_error:
                result.status = "skipped_low_sr"
                result.error = sr_error
                return result

            # Prepare output path
            out_dir = Path(self.out_root) / rel_subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(in_path).stem
            out_path = out_dir / f"{stem}.flac"
            result.output_path = str(out_path)

            # Skip if exists
            if out_path.exists():
                result.status = "skipped_exists"
                return result

            # Decode to temporary WAV
            tmp_wav = self._generate_temp_path("dec")
            ok, err = self.processor.decode_to_wav(in_path, tmp_wav)
            if not ok:
                result.error = f"Decode failed: {err}"
                return result

            # Measure initial peak
            peak, err = self.processor.measure_peak(tmp_wav)
            if err:
                result.error = f"Peak measurement failed: {err}"
                self._safe_remove(tmp_wav)
                return result

            result.detected_peak = peak
            work_wav = tmp_wav
            gain = 1.0

            # Handle clipping
            if peak >= 1.0:
                result.clipped_flag = True
                work_wav, peak, gain = self._handle_clipping(work_wav, peak)
                result.detected_peak = peak
                result.gain_applied = gain

            # Apply normalization
            work_wav, peak, gain = self._apply_normalization(work_wav, peak, gain)
            result.detected_peak = peak
            if gain != 1.0:
                result.gain_applied = gain

            # Final encode to FLAC
            ok, err = self.processor.encode_to_flac(work_wav, str(out_path))
            if not ok:
                result.error = f"Final encode failed: {err}"
                self._safe_remove(work_wav)
                return result

            # Cleanup
            self._safe_remove(work_wav)
            result.status = "ok"

        except Exception as e:
            result.error = f"Unexpected error: {e}"

        return result


class ConversionPipeline:
    """Main pipeline for batch audio conversion."""

    def __init__(
            self,
            inroot: str,
            outroot: str,
            workers: int = 4,
            normalize_db: Optional[float] = None,
            allow_boost: bool = False,
            min_sr: Optional[int] = None,
            input_csv: str = INPUT_CSV,
            success_csv: str = SUCCESS_CSV,
            failed_csv: str = FAILED_CSV
    ):
        self.inroot = Path(inroot)
        self.outroot = Path(outroot)
        self.workers = workers
        self.normalize_db = normalize_db
        self.allow_boost = allow_boost
        self.min_sr = min_sr
        self.input_csv = input_csv
        self.success_csv = success_csv
        self.failed_csv = failed_csv

    def _load_csv_records(self) -> list[dict]:
        """Load records from Stage2 CSV file."""
        if not os.path.exists(self.input_csv):
            logger.error(f"Input CSV not found: {self.input_csv}")
            return []

        df = pd.read_csv(self.input_csv)
        logger.info(f"Loaded {len(df)} records from {self.input_csv}")
        return df.to_dict('records')

    def _discover_files(self) -> list[tuple[str, str, dict]]:
        """Discover audio files matching CSV records."""
        csv_records = self._load_csv_records()
        if not csv_records:
            return []

        # Build a map of available files by XC ID
        # Filenames are like: xc422286_B.mp3, we extract the ID (422286)
        available_files = {}
        for root, _, filenames in os.walk(self.inroot):
            rel_dir = os.path.relpath(root, self.inroot)
            if rel_dir == ".":
                rel_dir = ""

            for fn in filenames:
                if not fn.startswith(".") and not fn.startswith("._"):
                    full_path = os.path.join(root, fn)
                    # Extract XC ID from filename (xc422286_B.mp3 -> 422286)
                    import re
                    match = re.match(r'xc(\d+)', fn.lower())
                    if match:
                        xc_id = match.group(1)
                        available_files[xc_id] = (full_path, rel_dir)

        # Match CSV records to files, ignoring non-sound files
        files = []
        sound_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

        for record in csv_records:
            file_id = str(record['id'])
            if file_id in available_files:
                full_path, rel_dir = available_files[file_id]
                # Check if it's a sound file
                if Path(full_path).suffix.lower() in sound_extensions:
                    files.append((full_path, rel_dir, record))
                else:
                    logger.debug(f"Skipping non-sound file: {full_path}")

        logger.info(f"Found {len(files)} sound files matching CSV records")
        return files

    def run(self):
        """Execute the conversion pipeline."""
        # Discover files
        files = self._discover_files()
        logger.info(f"Discovered {len(files)} files under {self.inroot}")

        # Setup
        self.outroot.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="audio_conv_")
        logger.info(f"Using temporary directory: {tmp_dir}")

        results = []

        try:
            # Create converter
            converter = FileConverter(
                str(self.outroot),
                tmp_dir,
                self.normalize_db,
                self.allow_boost,
                self.min_sr
            )

            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(converter.convert, path, rel, record): (path, rel, record)
                    for path, rel, record in files
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    if result.status == "ok":
                        logger.info(
                            f"✓ {result.input_path} → {result.output_path} "
                            f"(peak={result.detected_peak:.4f}, gain={result.gain_applied})"
                        )
                    else:
                        logger.warning(
                            f"x {result.input_path} "
                            f"(status={result.status}, error={result.error})"
                        )

            # Separate successful and failed conversions
            successful_records = []
            failed_records = []

            for result in results:
                if result.csv_record is not None:
                    if result.status == "ok":
                        successful_records.append(result.csv_record)
                    else:
                        # Add error information to failed record
                        failed_record = result.csv_record.copy()
                        failed_record['error'] = result.error
                        failed_record['status'] = result.status
                        failed_records.append(failed_record)

            # Save successful conversions
            if successful_records:
                df_success = pd.DataFrame(successful_records)
                df_success.to_csv(self.success_csv, index=False)
                logger.info(f"Wrote {len(successful_records)} successful records to {self.success_csv}")
            else:
                logger.warning("No successful conversions to save")

            # Save failed conversions
            if failed_records:
                df_failed = pd.DataFrame(failed_records)
                df_failed.to_csv(self.failed_csv, index=False)
                logger.info(f"Wrote {len(failed_records)} failed records to {self.failed_csv}")
            else:
                logger.info("No failed conversions")

        finally:
            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_sample_rate(value: str) -> int:
    """Parse sample rate from string (supports aliases like '16k', '24k')."""
    value = value.lower().strip()
    aliases = {"16k": 16000, "24k": 24000, "44k": 44100}

    if value in aliases:
        return aliases[value]

    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid sample rate: '{value}'. Use '16k', '24k', '44k', or an integer."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Convert audio files to mono 16kHz FLAC with clipping detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic conversion (MP3 to 16kHz FLAC)
  python Stage3_convert_mp3_to_16k_flac.py --inroot /path/to/mp3s \\
    --outroot /path/to/flacs

  # With parallel workers
  python Stage3_convert_mp3_to_16k_flac.py --inroot /path/to/mp3s \\
    --outroot /path/to/flacs --workers 8

  # With normalization
  python Stage3_convert_mp3_to_16k_flac.py --inroot /path/to/mp3s \\
    --outroot /path/to/flacs --normalize-db -1.0 --allow-boost
        """
    )

    # Required arguments
    parser.add_argument(
        "--inroot",
        required=True,
        metavar="DIR",
        help="Input root directory containing audio files"
    )
    parser.add_argument(
        "--outroot",
        required=True,
        metavar="DIR",
        help="Output root directory for converted FLAC files"
    )

    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--normalize-db",
        type=float,
        metavar="DBFS",
        help="Peak normalize to this dBFS (e.g., -1.0)"
    )
    parser.add_argument(
        "--allow-boost",
        action="store_true",
        help="Allow gain > 1.0 during normalization"
    )
    parser.add_argument(
        "--min-sr",
        type=parse_sample_rate,
        default=DEFAULT_MIN_SR,
        metavar="HZ",
        help="Minimum native sample rate (e.g., '24k', '16000')"
    )

    # CSV options
    parser.add_argument(
        "--input-csv",
        default=INPUT_CSV,
        metavar="FILE",
        help=f"Input CSV from Stage 2 (default: {INPUT_CSV})"
    )
    parser.add_argument(
        "--success-csv",
        default=SUCCESS_CSV,
        metavar="FILE",
        help=f"Output CSV for successful conversions (default: {SUCCESS_CSV})"
    )
    parser.add_argument(
        "--failed-csv",
        default=FAILED_CSV,
        metavar="FILE",
        help=f"Output CSV for failed conversions (default: {FAILED_CSV})"
    )

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 60)
    logger.info("Audio Conversion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input root:      {args.inroot}")
    logger.info(f"Output root:     {args.outroot}")
    logger.info(f"Input CSV:       {args.input_csv}")
    logger.info(f"Success CSV:     {args.success_csv}")
    logger.info(f"Failed CSV:      {args.failed_csv}")
    logger.info(f"Workers:         {args.workers}")
    logger.info(f"Normalize dB:    {args.normalize_db}")
    logger.info(f"Allow boost:     {args.allow_boost}")
    logger.info(f"Min sample rate: {args.min_sr}")
    logger.info("=" * 60)

    # Run pipeline
    pipeline = ConversionPipeline(
        inroot=args.inroot,
        outroot=args.outroot,
        workers=args.workers,
        normalize_db=args.normalize_db,
        allow_boost=args.allow_boost,
        min_sr=args.min_sr,
        input_csv=args.input_csv,
        success_csv=args.success_csv,
        failed_csv=args.failed_csv
    )

    pipeline.run()
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()