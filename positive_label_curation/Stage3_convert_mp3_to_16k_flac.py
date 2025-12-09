#!/usr/bin/env python3
"""
Audio Conversion Pipeline

Converts audio files to mono 16kHz 16-bit FLAC format with:
- Clipping detection and correction
- Optional peak normalization
- Minimum sample rate filtering
- Parallel processing
- Comprehensive logging

Usage:
    python convert_to_16k_flac.py --inroot ./raw --outroot ./processed --min-sr 16k
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
LOG_CSV = "conversion_log.csv"

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

    def convert(self, in_path: str, rel_subdir: str) -> ConversionResult:
        """Convert a single audio file through the complete pipeline."""
        result = ConversionResult(input_path=in_path)

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
            log_csv_path: str = LOG_CSV
    ):
        self.inroot = Path(inroot)
        self.outroot = Path(outroot)
        self.workers = workers
        self.normalize_db = normalize_db
        self.allow_boost = allow_boost
        self.min_sr = min_sr
        self.log_csv_path = log_csv_path

    def _discover_files(self) -> list[tuple[str, str]]:
        """Discover all audio files in input directory."""
        files = []
        for root, _, filenames in os.walk(self.inroot):
            rel_dir = os.path.relpath(root, self.inroot)
            if rel_dir == ".":
                rel_dir = ""

            for fn in filenames:
                if not fn.startswith("."):
                    full_path = os.path.join(root, fn)
                    files.append((full_path, rel_dir))

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
                    executor.submit(converter.convert, path, rel): (path, rel)
                    for path, rel in files
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

            # Save results
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(self.log_csv_path, index=False)
            logger.info(f"Wrote conversion log to {self.log_csv_path}")

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
        description="Convert audio files to mono 16kHz FLAC with clipping detection"
    )
    parser.add_argument(
        "--inroot",
        required=True,
        help="Input root directory"
    )
    parser.add_argument(
        "--outroot",
        required=True,
        help="Output root directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--normalize-db",
        type=float,
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
        help="Minimum native sample rate (e.g., '24k', '16000')"
    )
    parser.add_argument(
        "--log-csv",
        default=LOG_CSV,
        help=f"Output CSV log path (default: {LOG_CSV})"
    )

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 60)
    logger.info("Audio Conversion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input root:      {args.inroot}")
    logger.info(f"Output root:     {args.outroot}")
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
        log_csv_path=args.log_csv
    )

    pipeline.run()
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()