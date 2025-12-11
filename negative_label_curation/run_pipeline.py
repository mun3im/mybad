#!/usr/bin/env python3
"""
Negative sample extraction pipeline for bird detection dataset.

Target: 25,000 negative samples

Pipeline:
1. Extract negatives from DCASE (BirdVox, Freefield1010, Warblrb10k)
2. Extract negatives from ESC50
3. Extract negatives from FSC22
4. Calculate remaining needed from DATASEC and extract dynamically
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_orchestrator.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Target counts
TARGET_NEGATIVE = 25000

# Music vs Voice allocation for DATASEC (minimal music for field deployment)
MUSIC_RATIO = 0.05  # 5% music, 95% voices

# Output directories
NEGATIVE_DIRS = {
    'esc': '/Volumes/Evo/mybad2/negative/esc',
    'fsc': '/Volumes/Evo/mybad2/negative/fsc',
    'dcase_bv': '/Volumes/Evo/mybad2/negative/bv',
    'dcase_ff': '/Volumes/Evo/mybad2/negative/ff',
    'dcase_wb': '/Volumes/Evo/mybad2/negative/wb',
    'datasec': '/Volumes/Evo/mybad2/negative/datasec'
}


def count_wav_files(directory):
    """Count .wav files in a directory."""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob("*.wav")))


def update_stage4_config(target_samples, music_ratio):
    """Update Stage4 script with dynamic target and music ratio."""
    stage4_path = Path("Stage4_extract_datasec.py")

    if not stage4_path.exists():
        logger.error("Stage4_extract_datasec.py not found!")
        return False

    content = stage4_path.read_text()

    # Update TARGET_NEGATIVE_TOTAL
    content = re.sub(
        r'TARGET_NEGATIVE_TOTAL = \d+',
        f'TARGET_NEGATIVE_TOTAL = {target_samples}',
        content
    )

    # Update music allocation ratio in the allocation section
    # Find the line with music_allocation = int(remaining * 0.25)
    content = re.sub(
        r'music_allocation = int\(remaining \* (0\.\d+)\)',
        f'music_allocation = int(remaining * {music_ratio})',
        content
    )

    # Update the comment as well
    content = re.sub(
        r'# (\d+)% for music',
        f'# {int(music_ratio*100)}% for music',
        content
    )
    content = re.sub(
        r'# (\d+)% for voices',
        f'# {int((1-music_ratio)*100)}% for voices',
        content
    )

    stage4_path.write_text(content)
    logger.info(f"✓ Updated Stage4: target={target_samples}, music_ratio={music_ratio:.1%}")
    return True


def run_stage(script_name, description):
    """Run a stage script and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running {description}")
    logger.info(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
        logger.info(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {description} failed!")
        logger.error(f"Return code: {e.returncode}")
        return False


def print_stage_summary(stage_name, count, cumulative):
    """Print formatted stage summary."""
    logger.info(f"\n{'─'*70}")
    logger.info(f"STAGE COMPLETE: {stage_name}")
    logger.info(f"  Samples from this stage: {count:,}")
    logger.info(f"  Cumulative total:        {cumulative:,} / {TARGET_NEGATIVE:,}")
    logger.info(f"  Remaining needed:        {max(0, TARGET_NEGATIVE - cumulative):,}")
    logger.info(f"{'─'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 25,000 negative samples for bird detection dataset"
    )
    parser.add_argument(
        "--music-ratio",
        type=float,
        default=MUSIC_RATIO,
        help=f"Ratio of music samples in DATASEC allocation (default: {MUSIC_RATIO})"
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 (DCASE) - use existing files"
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 (ESC50) - use existing files"
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Skip Stage 3 (FSC22) - use existing files"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("NEGATIVE SAMPLE EXTRACTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Target: {TARGET_NEGATIVE:,} negative samples")
    logger.info(f"Music ratio for DATASEC: {args.music_ratio:.1%}")
    logger.info("")

    cumulative = 0

    # ========================== STAGE 1: DCASE (NEGATIVE) ==========================
    logger.info("STAGE 1: DCASE datasets (BirdVox, Freefield1010, Warblrb10k)")
    logger.info("         Extracting NEGATIVE samples only")

    if not args.skip_stage1:
        if not run_stage("Stage1_extract_dcase.py", "Stage 1: DCASE Negative"):
            logger.error("Pipeline failed at Stage 1")
            return 1
    else:
        logger.info("(Skipped - using existing files)")

    dcase_negative = sum([
        count_wav_files(NEGATIVE_DIRS['dcase_bv']),
        count_wav_files(NEGATIVE_DIRS['dcase_ff']),
        count_wav_files(NEGATIVE_DIRS['dcase_wb'])
    ])
    cumulative += dcase_negative
    print_stage_summary("DCASE (BV/FF/WB)", dcase_negative, cumulative)

    # ========================== STAGE 2: ESC50 (NEGATIVE) ==========================
    logger.info("\nSTAGE 2: ESC50 dataset - Extracting NEGATIVE samples only")

    if not args.skip_stage2:
        if not run_stage("Stage2_extract_esc50.py", "Stage 2: ESC50 Negative"):
            logger.error("Pipeline failed at Stage 2")
            return 1
    else:
        logger.info("(Skipped - using existing files)")

    esc_negative = count_wav_files(NEGATIVE_DIRS['esc'])
    cumulative += esc_negative
    print_stage_summary("ESC50", esc_negative, cumulative)

    # ========================== STAGE 3: FSC22 (NEGATIVE) ==========================
    logger.info("\nSTAGE 3: FSC22 dataset - Extracting NEGATIVE samples only")

    if not args.skip_stage3:
        if not run_stage("Stage3_extract_fsc22.py", "Stage 3: FSC22 Negative"):
            logger.error("Pipeline failed at Stage 3")
            return 1
    else:
        logger.info("(Skipped - using existing files)")

    fsc_negative = count_wav_files(NEGATIVE_DIRS['fsc'])
    cumulative += fsc_negative
    print_stage_summary("FSC22", fsc_negative, cumulative)

    # ========================== STAGE 4: DATASEC (DYNAMIC ALLOCATION) ==========================
    datasec_needed = TARGET_NEGATIVE - cumulative

    logger.info(f"\n{'='*70}")
    logger.info("STAGE 4: DATASEC - DYNAMIC ALLOCATION")
    logger.info(f"{'='*70}")
    logger.info(f"Samples collected so far: {cumulative:,} / {TARGET_NEGATIVE:,}")
    logger.info(f"Need from DATASEC:        {datasec_needed:,}")
    logger.info("")

    if datasec_needed <= 0:
        logger.warning(f"⚠ Already have {cumulative:,} negative samples!")
        logger.warning(f"   No DATASEC extraction needed (target is {TARGET_NEGATIVE:,})")
    else:
        logger.info("DATASEC Extraction Strategy:")
        logger.info("  1. Quality filter all clips (remove quiet, clipped, low dynamic range)")
        logger.info("  2. Keep ALL samples from small folders")
        logger.info("  3. From remaining quota, allocate:")

        music_samples = int(datasec_needed * args.music_ratio)
        voice_samples = datasec_needed - music_samples

        logger.info(f"     - Music:  ~{music_samples:,} samples ({args.music_ratio:.1%}) - minimal, unlikely in field")
        logger.info(f"     - Voices: ~{voice_samples:,} samples ({1-args.music_ratio:.1%}) - realistic for field deployment")
        logger.info("")

        # Update Stage4 config
        if not update_stage4_config(datasec_needed, args.music_ratio):
            logger.error("Failed to update Stage4 configuration")
            return 1

        logger.info(f"Running Stage 4 with target: {datasec_needed:,} samples...")

        if not run_stage("Stage4_extract_datasec.py", f"Stage 4: DATASEC ({datasec_needed:,} samples)"):
            logger.error("Pipeline failed at Stage 4")
            return 1

        datasec_actual = count_wav_files(NEGATIVE_DIRS['datasec'])
        cumulative += datasec_actual
        print_stage_summary("DATASEC", datasec_actual, cumulative)

    # ========================== FINAL SUMMARY ==========================
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE - FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info("")
    logger.info("Negative samples by source:")
    logger.info(f"  ESC50:             {esc_negative:,}")
    logger.info(f"  FSC22:             {fsc_negative:,}")
    logger.info(f"  DCASE (BV/FF/WB):  {dcase_negative:,}")

    if datasec_needed > 0:
        datasec_actual = count_wav_files(NEGATIVE_DIRS['datasec'])
        logger.info(f"  DATASEC:           {datasec_actual:,} (target: {datasec_needed:,})")

    logger.info(f"  {'─'*40}")
    logger.info(f"  TOTAL:             {cumulative:,} / {TARGET_NEGATIVE:,}")
    logger.info("")

    if cumulative >= TARGET_NEGATIVE:
        logger.info(f"✓ SUCCESS: Reached target of {TARGET_NEGATIVE:,} negative samples!")
        if cumulative > TARGET_NEGATIVE:
            logger.info(f"  ({cumulative - TARGET_NEGATIVE:,} samples over target)")
    else:
        shortage = TARGET_NEGATIVE - cumulative
        logger.warning(f"⚠ SHORT: Need {shortage:,} more samples to reach {TARGET_NEGATIVE:,} target")
        logger.warning(f"  This may be due to quality filtering removing more samples than expected")

    logger.info(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
