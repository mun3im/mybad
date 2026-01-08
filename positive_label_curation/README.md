# Xeno-Canto Bird Audio Dataset Curation Pipeline

A complete 7-stage pipeline for fetching, processing, and curating balanced bird audio datasets from Xeno-Canto (xeno-canto.org). Part of the Malaysian Bird Audio Dataset (MyBAD) project.

## Pipeline Overview

```
Stage 1: Fetch Metadata → Stage 2: Download MP3s → Stage 3: Convert to FLAC →
Stage 4: Deduplicate → Stage 5: Extract Clips → Stage 6: Balance Species →
Stage 7: Create Dataset
```

## Quick Start

### One-Command Pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

The shell script runs all 7 stages automatically with optimized settings:
- Fetches Southeast Asian bird metadata
- Downloads ~40,000 MP3 files
- Converts to 16kHz mono FLAC
- Deduplicates recordings
- Extracts 3-second clips
- Balances to 20,000 clips
- Creates final dataset directory

### Manual Stage-by-Stage

```bash
# Stage 1: Fetch metadata
python Stage1_xc_fetch_bird_metadata.py --country all

# Stage 2: Download MP3s
python Stage2_xc_download_bird_mp3s.py --outroot /Volumes/Evo/xc-asean-mp3s

# Stage 3: Convert to FLAC
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /Volumes/Evo/xc-asean-mp3s \
  --outroot /Volumes/Evo/xc-asean-flac \
  --workers 8

# Stage 4: Deduplicate
python Stage4_find_flac_duplicates.py /Volumes/Evo/xc-asean-flac

# Stage 5: Extract clips
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /Volumes/Evo/xc-asean-flac \
  --outroot /Volumes/Evo/xc-asean-clips \
  --no-quarantine

# Stage 6: Balance species
python Stage6_balance_species.py --outroot /Volumes/Evo/xc-asean-clips

# Stage 7: Create final dataset
python Stage7_move_to_quarantine.py \
  --outroot /Volumes/Evo/xc-asean-clips \
  --dataset-dir /Volumes/Evo/newdataset
```

## Features

- **Automated workflow**: Run all 7 stages with a single shell script
- **Resume-safe**: Skips existing files and CSVs, safe to re-run
- **Quality filtering**: Removes duplicates, low-quality recordings, and non-sound files
- **Species balancing**: Creates ecologically diverse datasets with configurable target size
- **Rich metadata**: Preserves all Xeno-Canto metadata (species, location, recorder, quality, etc.)
- **Parallel processing**: Multi-threaded conversion and processing
- **Audio deduplication**: Detects both perfect and near-duplicates using mel-spectrogram embeddings

## Requirements

### Python Dependencies

```bash
pip install pandas requests librosa soundfile pydub tqdm scikit-learn matplotlib
```

### System Dependencies

- **ffmpeg**: Required for audio conversion
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

### Python Version

- Python 3.8 or higher

## Pipeline Stages

### Stage 1: Fetch Metadata

**Script**: `Stage1_xc_fetch_bird_metadata.py`

Fetches bird recording metadata from Xeno-Canto API for Southeast Asian countries.

**Usage**:
```bash
# Fetch all Southeast Asian countries (default)
python Stage1_xc_fetch_bird_metadata.py --country all

# Fetch single country
python Stage1_xc_fetch_bird_metadata.py --country Malaysia
```

**Output**: `Stage1_xc_sea_birds.csv`

**Countries**: Brunei, Cambodia, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, Thailand, Timor-Leste, Vietnam

**CSV Schema**:
- `id` - Xeno-Canto recording ID
- `en` - English species name
- `rec` - Recordist name
- `cnt` - Country
- `lat`, `lon` - GPS coordinates
- `file` - Download URL
- `lic` - License URL
- `q` - Quality rating (A/B/C/D/E/n)
- `length` - Recording duration (MM:SS)
- `smp` - Sample rate

---

### Stage 2: Download MP3s

**Script**: `Stage2_xc_download_bird_mp3s.py`

Downloads audio files from Xeno-Canto, organized by species.

**Usage**:
```bash
python Stage2_xc_download_bird_mp3s.py --outroot /path/to/mp3s

# Test with first 100 files
python Stage2_xc_download_bird_mp3s.py --outroot /path/to/mp3s --limit 100
```

**Features**:
- Skips recordings < 3 seconds
- Validates file sizes (minimum 1 KB)
- Handles rate limiting (0.1s delay between requests)
- Resume-safe (skips existing files)
- Organized by species folders
- Automatic retry with exponential backoff

**Outputs**:
- `Stage2_xc_successful_downloads.csv` - Successfully downloaded files
- `Stage2_xc_failed_downloads.csv` - Failed downloads for retry
- `Stage2_download_log.csv` - Detailed download log

**Filename format**: `xc{id}_{quality}.mp3` (e.g., `xc422286_A.mp3`)

**Important**: This script was recently fixed to prevent duplicate CSV entries on reruns.

---

### Stage 3: Convert to FLAC

**Script**: `Stage3_convert_mp3_to_16k_flac.py`

Converts MP3s to 16kHz mono FLAC format for consistency.

**Usage**:
```bash
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /path/to/mp3s \
  --outroot /path/to/flacs \
  --workers 8
```

**Features**:
- Parallel processing (default: 8 workers)
- Preserves directory structure
- Matches files to CSV records by XC ID
- Skips non-sound files
- Optional normalization (`--normalize-db`, `--allow-boost`)
- Resume-safe (skips existing FLAC files)

**Outputs**:
- `Stage3_xc_successful_conversion.csv` - Successful conversions with metadata
- `Stage3_failed_conversion.csv` - Failed conversions

**Filename format**: `xc{id}_{quality}.flac`

---

### Stage 4: Deduplicate Files

**Script**: `Stage4_find_flac_duplicates.py`

Detects and quarantines duplicate recordings using audio similarity.

**Usage**:
```bash
# With default settings (quarantines both perfect and near-duplicates)
python Stage4_find_flac_duplicates.py /path/to/flacs

# Quarantine only perfect duplicates
python Stage4_find_flac_duplicates.py /path/to/flacs --no-quarantine-near-duplicates

# Dry run to preview
python Stage4_find_flac_duplicates.py /path/to/flacs --dry-run
```

**Deduplication Algorithm**:

1. **MD5 Hash**: Perfect duplicate detection
2. **Mel-spectrogram Embeddings**: Acoustic similarity (128 mel bins, 8kHz max freq)
3. **Cosine Similarity**: Threshold = 0.95 for near-duplicates
4. **Quarantine Strategy**:
   - Perfect duplicates (≥ 0.999): Quarantine newer recording (higher XC ID)
   - Near-duplicates (≥ 0.95): Optional quarantine (e.g., different bitrates)

**Outputs**:
- `Stage4_unique_flacs.csv` - Metadata for non-quarantined files
- `quarantine/` - Directory containing duplicate files
- `duplicate_pairs.txt` - Near-duplicates for manual review

**Important**: Use `--no-quarantine-near-duplicates` if you want to keep files at different bitrates.

---

### Stage 5: Extract 3-Second Clips

**Script**: `Stage5_extract_3s_clips_from_flac.py`

Extracts multiple 3-second clips from each recording.

**Usage**:
```bash
# Balanced workflow (for Stage 6)
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /path/to/flacs \
  --outroot /path/to/clips \
  --no-quarantine
```

**Features**:
- Extracts multiple non-overlapping clips per file
- RMS threshold: 0.001 (avoids silence)
- Clipping detection and soft limiting
- Preserves all Stage 4 metadata
- Adds `start_ms` and `clip_filename` fields

**Output**: `Stage5_unique_3sclips.csv`

**CSV includes**:
- All metadata from Stage 4 (species, recorder, country, coordinates, license, quality)
- `start_ms` - Clip start time in milliseconds
- `clip_filename` - Output filename

**Filename format**: `xc{id}_{quality}_{start_ms}.wav` (e.g., `xc422286_A_0.wav`, `xc422286_A_3000.wav`)

**Important**: Use `--no-quarantine` flag when running before Stage 6 to keep all clips for balancing.

---

### Stage 6: Balance Species

**Script**: `Stage6_balance_species.py`

Balances species representation to create ecologically diverse dataset.

**Usage**:
```bash
# Use default target size (20,000)
python Stage6_balance_species.py --outroot /path/to/clips

# Custom target size
python Stage6_balance_species.py --outroot /path/to/clips --target-size 50000
```

**Balancing Strategy**:
- Equal representation across species
- Random sampling within species
- Prioritizes diversity and quality
- Gini coefficient tracking for distribution equality

**Target Size**: Default 20,000 clips (configurable via `--target-size` or `TARGET_DATASET_SIZE` constant in script)

**Output**: `Stage6_balanced_clips.csv` - Metadata listing which clips to keep

**Important**: This is a metadata-only operation. Files are not moved until Stage 7.

---

### Stage 7: Create Final Dataset

**Script**: `Stage7_move_to_quarantine.py`

Creates final dataset directory with balanced clips from Stage 6.

**Usage**:
```bash
# Auto-generated directory name
python Stage7_move_to_quarantine.py --outroot /path/to/clips

# Custom dataset directory
python Stage7_move_to_quarantine.py \
  --outroot /path/to/clips \
  --dataset-dir /path/to/my-dataset

# Dry run to preview
python Stage7_move_to_quarantine.py --outroot /path/to/clips --dry-run
```

**Features**:
- Auto-increments directory names: `dataset_20000_001`, `dataset_20000_002`, etc.
- Copies selected files to dataset directory
- Creates dataset manifest CSV
- Preserves original clips in source directory
- Verifies final count matches expected

**Outputs**:
- `dataset_{size}_{counter}/` - Final dataset directory
- `dataset_manifest.csv` - Metadata for all files in dataset

**Directory naming**: Format is `dataset_{target_size}_{counter:03d}` where counter auto-increments if directory exists.

---

## CSV Data Flow

```
Stage1_xc_sea_birds.csv
  ├─> id, en, rec, cnt, lat, lon, file, lic, q, length, smp
  ↓
Stage2_xc_successful_downloads.csv
  ├─> id, en, rec, cnt, lat, lon, lic, q, length, smp
  ↓
Stage3_xc_successful_conversion.csv (adds conversion metadata)
  ├─> Same as Stage2 + conversion fields
  ↓
Stage4_unique_flacs.csv (filtered: no duplicates)
  ├─> Same as Stage3 (deduplicated)
  ↓
Stage5_unique_3sclips.csv
  ├─> All Stage4 fields + start_ms, clip_filename
  ↓
Stage6_balanced_clips.csv
  ├─> Same as Stage5 (balanced subset)
  ↓
Stage7: dataset_manifest.csv (final dataset)
  └─> Same as Stage6 (physically copied to dataset dir)
```

## Output Directory Structure

```
project/
├── Stage1_xc_sea_birds.csv
├── Stage2_xc_successful_downloads.csv
├── Stage2_download_log.csv
├── Stage2_xc_failed_downloads.csv
├── Stage3_xc_successful_conversion.csv
├── Stage3_failed_conversion.csv
├── Stage4_unique_flacs.csv
├── Stage5_unique_3sclips.csv
├── Stage6_balanced_clips.csv
├── run_pipeline.sh
├── Stage1_xc_fetch_bird_metadata.py
├── Stage2_xc_download_bird_mp3s.py
├── Stage3_convert_mp3_to_16k_flac.py
├── Stage4_find_flac_duplicates.py
├── Stage5_extract_3s_clips_from_flac.py
├── Stage6_balance_species.py
├── Stage7_move_to_quarantine.py
│
├── /Volumes/Evo/xc-asean-mp3s/          # Stage 2 output
│   ├── Species_Name_1/
│   │   ├── xc123456_A.mp3
│   │   └── xc123457_B.mp3
│   └── Species_Name_2/
│       └── xc123458_A.mp3
│
├── /Volumes/Evo/xc-asean-flac/          # Stage 3 output
│   ├── Species_Name_1/
│   │   ├── xc123456_A.flac
│   │   └── xc123457_B.flac
│   └── quarantine/                      # Stage 4 duplicates
│       └── xc123459_A.flac
│
├── /Volumes/Evo/xc-asean-clips/         # Stage 5 output
│   ├── xc123456_A_0.wav
│   ├── xc123456_A_3000.wav
│   ├── xc123456_A_6000.wav
│   └── xc123457_B_0.wav
│
└── /Volumes/Evo/newdataset/             # Stage 7 output
    └── dataset_20000_001/
        ├── dataset_manifest.csv
        ├── xc123456_A_0.wav
        ├── xc123457_B_0.wav
        └── ... (20,000 files)
```

## Configuration

### Pipeline Configuration (run_pipeline.sh)

Edit paths in `run_pipeline.sh`:

```bash
MP3_DIR="/Volumes/Evo/xc-asean-mp3s"
FLAC_DIR="/Volumes/Evo/xc-asean-flac"
CLIPS_DIR="/Volumes/Evo/xc-asean-clips"
DATASET_DIR="/Volumes/Evo/newdataset"
```

### Target Dataset Size

**Method 1**: Edit `Stage6_balance_species.py`

```python
TARGET_DATASET_SIZE = 20000  # Change to desired size
```

**Method 2**: Use CLI argument

```bash
python Stage6_balance_species.py --target-size 50000
```

### Audio Processing Parameters

**Stage 3 - Conversion**:
- Sample rate: 16000 Hz
- Channels: Mono
- Format: FLAC

**Stage 4 - Deduplication**:
- Similarity threshold: 0.95
- Perfect duplicate threshold: 0.999
- Mel bins: 128
- Max frequency: 8000 Hz

**Stage 5 - Clip Extraction**:
- Clip duration: 3.0 seconds
- Minimum RMS: 0.001
- Minimum separation: 1.5 seconds

## Performance Notes

### Expected Processing Times

| Stage | Time (approx) | Notes |
|-------|---------------|-------|
| Stage 1 | 5-10 minutes | API-dependent |
| Stage 2 | 2-6 hours | 40k files, network-dependent |
| Stage 3 | 30-60 minutes | 8 workers |
| Stage 4 | 60-120 minutes | Audio similarity computation |
| Stage 5 | 20-40 minutes | Clip extraction |
| Stage 6 | < 1 minute | Metadata-only |
| Stage 7 | 10-20 minutes | File copying |

**Total pipeline runtime**: 4-8 hours

### Disk Space Requirements

- MP3s: ~15-20 GB
- FLACs: ~25-35 GB
- Clips: ~40-60 GB
- Final dataset (20,000 clips): ~2-3 GB

**Total working space**: ~80-120 GB

### File Counts (Southeast Asia)

- Stage 1: ~42,000 unique records
- Stage 2: ~39,000 successful downloads
- Stage 3: ~39,000 FLACs
- Stage 4: ~38,000 unique FLACs (after deduplication)
- Stage 5: ~200,000+ clips
- Stage 6: 20,000 balanced clips (default)
- Stage 7: 20,000 clips in final dataset

## Troubleshooting

### Stage 2: Download Failures

**Issue**: Some downloads fail with 404 or timeout errors

**Solution**:
- Failed downloads are logged in `Stage2_xc_failed_downloads.csv`
- Xeno-Canto occasionally removes recordings
- Re-run Stage 2 to retry failed downloads (it will skip existing files)

### Stage 2: Duplicate CSV Entries

**Issue**: Running Stage 2 multiple times created duplicate entries in CSV

**Solution**:
- This has been fixed in the latest version
- To clean existing duplicates:
  ```bash
  head -1 Stage2_xc_successful_downloads.csv > temp.csv
  tail -n +2 Stage2_xc_successful_downloads.csv | sort -u >> temp.csv
  mv temp.csv Stage2_xc_successful_downloads.csv
  ```

### Stage 3: Conversion Errors

**Issue**: FFmpeg errors or missing files

**Solution**:
- Ensure ffmpeg is installed: `ffmpeg -version`
- Check `Stage3_failed_conversion.csv` for specific errors
- Re-run Stage 3 (it will skip existing conversions)

### Stage 3: "Found 0 sound files"

**Issue**: Stage 3 reports no matching files despite files existing

**Solution**:
- This has been fixed (XC ID extraction from filenames)
- Ensure Stage 2 CSV is not corrupted
- Check that MP3 filenames match format: `xc{id}_{quality}.mp3`

### Stage 4: Out of Memory

**Issue**: Memory errors during embedding computation

**Solution**:
- Process in smaller batches
- Increase system RAM
- Use `--no-quarantine-near-duplicates` to skip similarity computation
- Close other applications

### Pipeline Interruption

**Issue**: Pipeline stopped mid-execution

**Solution**:
- The pipeline is resume-safe
- Simply re-run `./run_pipeline.sh`
- Each stage skips existing files and CSVs

## Advanced Usage

### Different Geographic Regions

Edit `Stage1_xc_fetch_bird_metadata.py`:

```python
COUNTRIES = {
    "United States": "US",
    "Canada": "CA",
    "Mexico": "MX"
}
```

### Quality Filtering

Edit `Stage2_xc_download_bird_mp3s.py` to only download high-quality recordings:

```python
# Only download A and B quality
if q_char not in ['A', 'B']:
    logger.info(f"Skipping quality {q_char}")
    continue
```

### Custom Clip Duration

Edit `Stage5_extract_3s_clips_from_flac.py`:

```python
CLIP_DURATION = 5.0  # 5-second clips instead of 3
```

### Species-Specific Dataset

Filter `Stage1_xc_sea_birds.csv` for specific species before running Stage 2:

```python
import pandas as pd

df = pd.read_csv('Stage1_xc_sea_birds.csv')
df_filtered = df[df['en'].isin(['Oriental Magpie-Robin', 'Common Tailorbird'])]
df_filtered.to_csv('Stage1_filtered.csv', index=False)
```

Then run pipeline with `--input-csv Stage1_filtered.csv` for Stage 2.

## Key Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sample Rate | 16 kHz | Consistent audio processing |
| Clip Duration | 3.0 sec | Fixed-length segments |
| Audio Format | FLAC (Stage 3-4), WAV (Stage 5+) | Lossless compression |
| Channels | Mono | Single-channel audio |
| Min Separation | 1.5 sec | Between clips from same file |
| Target Dataset | 20,000 | Balanced final dataset size |
| Workers | 8 | Parallel processing threads |

## Project Context

Part of the **Malaysian Bird Audio Dataset (MyBAD)** project for training bird sound classification models. The pipeline ensures high-quality, deduplicated training data with rich metadata from community-contributed Xeno-Canto recordings.

## Recent Updates

**2026-01-08: Complete Pipeline Overhaul**
- Added `run_pipeline.sh` for automated execution
- Standardized CSV naming: `Stage1_*.csv`, `Stage2_*.csv`, etc.
- Fixed Stage 2 duplicate CSV bug
- Fixed Stage 3 file matching (XC ID extraction)
- Updated Stage 7 to create dataset directories instead of quarantine
- Added auto-incrementing dataset directory naming
- Set default target dataset size to 20,000

**2025-12-23: Metadata Enrichment & CLI Standardization**
- Stage 2: Added metadata fields (recorder, country, lat/lon, license)
- Stage 4: Creates `Stage4_unique_flacs.csv` with metadata
- Stage 5: Enriches output with metadata from Stage 4
- CLI standardization across all stages

**2025-12-16: Species Balancing**
- Added Stage 6: Species balancing with diversity optimization
- Added Stage 7: Dataset directory creation

## Citation

If you use this pipeline, please cite Xeno-Canto:

```
Xeno-canto: Bird sounds from around the world.
https://www.xeno-canto.org/
```

## License

This pipeline is provided for research and educational purposes. Please respect Xeno-Canto's terms of service and the individual Creative Commons licenses of each recording.

## Contributing

Contributions welcome! Please open issues for bugs or feature requests.

## Repository

https://github.com/mun3im/mybad-curation/tree/main/positive-label-curation

## Support

For questions or issues, please open a GitHub issue with:
- Stage where error occurred
- Error message
- Relevant CSV/log files
- System information (OS, Python version, ffmpeg version)
