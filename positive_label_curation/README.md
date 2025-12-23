# Positive Label Curation Pipeline

Audio data curation pipeline for the MyBAD (Malaysian Bird Audio Dataset) project. This pipeline fetches, downloads, processes, and curates bird audio recordings from Xeno-Canto for Southeast Asian countries.

## Pipeline Overview

```
Stage 1: Fetch Metadata → Stage 2: Download MP3s → Stage 3: Convert to FLAC →
Stage 4: Detect Duplicates → Stage 5: Extract Clips → Stage 6: Balance Species →
Stage 7: Move to Quarantine
```

## Pipeline Stages

### Stage 1: Fetch Metadata
**Script:** `Stage1_xc_fetch_bird_metadata.py`

Fetches bird species metadata from Xeno-Canto API for Southeast Asian countries (Malaysia, Singapore, Indonesia, Brunei, Thailand).

**Prerequisites:**
```bash
export XENO_API_KEY="your_xeno_canto_api_key"
```

**Usage:**
```bash
# Fetch all Southeast Asian countries (RECOMMENDED)
python Stage1_xc_fetch_bird_metadata.py --country all --output-csv xc_sea_birds.csv

# Fetch single country
python Stage1_xc_fetch_bird_metadata.py --country Malaysia --output-csv xc_my_birds.csv

# Fetch specific countries
python Stage1_xc_fetch_bird_metadata.py --countries Malaysia Indonesia Thailand --output-csv xc_multi.csv
```

**Output:**
- `Stage1_xc_sea_birds.csv` - Metadata for all bird recordings
- Automatic deduplication by XC ID
- Country breakdown statistics

---

### Stage 2: Download Audio
**Script:** `Stage2_xc_download_bird_mp3s.py`

Downloads MP3 audio files from Xeno-Canto based on metadata CSV.

**Features:**
- Automatic retry with exponential backoff
- Skips files < 3 seconds
- Skips 4xx/5xx HTTP errors immediately (no retries)
- Creates metadata CSV with recorder, country, coordinates, license

**Usage:**
```bash
# Download all files from Stage1 CSV
python Stage2_xc_download_bird_mp3s.py \
  --input-csv Stage1_xc_sea_birds.csv \
  --outroot /Volumes/Evo/xc-asean-mp3s

# Test with first 100 rows
python Stage2_xc_download_bird_mp3s.py \
  --input-csv Stage1_xc_sea_birds.csv \
  --outroot /path/to/output \
  --limit 100
```

**Output:**
- MP3 files organized by species: `{species_name}/xc{id}_{quality}.mp3`
- `Stage2_xc_successful_downloads.csv` - Successful downloads with metadata:
  - `id, en, rec, cnt, lat, lon, lic, q, length, smp`
- `Stage2_download_log.csv` - Complete download log with status/errors
- `Stage2_xc_failed_downloads.csv` - Failed downloads for retry

---

### Stage 3: Convert to FLAC
**Script:** `Stage3_convert_mp3_to_16k_flac.py`

Converts MP3 files to 16kHz mono FLAC format for consistent processing.

**Usage:**
```bash
# Basic conversion
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /path/to/mp3s \
  --outroot /path/to/flacs

# With parallel workers and normalization
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /path/to/mp3s \
  --outroot /path/to/flacs \
  --workers 8 \
  --normalize-db -1.0 \
  --allow-boost
```

**Output:**
- FLAC files (16kHz, mono): `{species_name}/xc{id}_{quality}.flac`
- Conversion log CSV

---

### Stage 4: Detect Duplicates
**Script:** `Stage4_find_flac_duplicates.py`

Detects and quarantines duplicate audio clips using acoustic similarity. **Quarantines newer duplicates** (higher XC numbers) to keep older recordings.

**Features:**
- Computes mel-spectrogram embeddings for each audio file
- Performs all-pairs similarity comparison using cosine similarity
- Automatically quarantines perfect duplicates (similarity ≥ 0.999)
- **Creates Stage4_unique_flacs.csv** - CSV of non-quarantined files with metadata

**Usage:**
```bash
# Find duplicates and create unique files CSV
python Stage4_find_flac_duplicates.py /path/to/flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --stage4-csv Stage4_unique_flacs.csv

# Dry run to preview
python Stage4_find_flac_duplicates.py /path/to/flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --dry-run
```

**Output:**
- `duplicate_pairs.txt` - Near-duplicates for manual review
- `quarantine/` - Newer duplicate files moved here
- `Stage4_unique_flacs.csv` - **Metadata for kept files** (same columns as Stage2)

**See:** [Stage4_duplicate_detection_algorithm_documentation.md](./Stage4_duplicate_detection_algorithm_documentation.md) for detailed algorithm documentation.

---

### Stage 5: Extract Clips
**Script:** `Stage5_extract_3s_clips_from_flac.py`

Extracts 3-second non-overlapping clips from full-length FLAC recordings with **metadata enrichment**.

**Features:**
- RMS-based clip selection (threshold: 0.001)
- Clipping detection and soft limiting
- **Metadata enrichment** from Stage4 CSV (recorder, country, coordinates, license)
- Dual workflow: RMS-only OR balanced (for Stage6)

**Usage:**
```bash
# Balanced workflow (RECOMMENDED - keeps all clips for Stage6)
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /path/to/flacs \
  --outroot /path/to/clips \
  --output-csv clips_log.csv \
  --metadata-csv Stage4_unique_flacs.csv \
  --threshold 0.001 \
  --no-quarantine

# RMS-only workflow (immediate dataset, no Stage6)
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /path/to/flacs \
  --outroot /path/to/clips \
  --output-csv clips_log.csv \
  --metadata-csv Stage4_unique_flacs.csv \
  --max-clips 25000
```

**Output:**
- WAV clips (16kHz, PCM_16, 3 seconds): `xc{id}_{quality}_{start_ms}.wav`
- `clips_log.csv` with metadata:
  - Core: `xc_id, species, quality`
  - **Metadata** (from Stage4): `recorder, country, latitude, longitude, license`
  - Source: `source_file, source_duration_sec, original_length, sample_rate`
  - Clip: `clip_start_ms, clip_duration_sec, rms_energy, was_clipped`
  - Output: `out_filename, out_path`

**Important:** Use `--no-quarantine` flag when running before Stage 6 to keep all clips for balancing.

---

### Stage 6: Balance Species
**Script:** `Stage6_balance_species.py`

Applies species-level undersampling to achieve ecological diversity while maximizing recording variety and quality.

**Usage:**
```bash
python Stage6_balance_species.py \
  --input-csv clips_log.csv \
  --outroot /path/to/clips \
  --output-csv balanced_clips.csv \
  --target-size 25000 \
  --plots species_balance.png
```

**Key Features:**
- Diversity-aware sampling (prioritizes unique XC recordings)
- Quality-weighted selection (A > B > C > D ratings)
- Gini coefficient tracking for distribution equality
- Generates before/after distribution plots

**Output:**
- `balanced_clips.csv` - Metadata listing which clips to keep
- `species_balance.png` - Before/after distribution visualization

---

### Stage 7: Move to Quarantine
**Script:** `Stage7_move_to_quarantine.py`

Moves files excluded from the balanced dataset to a quarantine subdirectory.

**Usage:**
```bash
python Stage7_move_to_quarantine.py \
  --input-csv balanced_clips.csv \
  --outroot /path/to/clips

# Preview what would be moved
python Stage7_move_to_quarantine.py \
  --input-csv balanced_clips.csv \
  --outroot /path/to/clips \
  --dry-run
```

**Purpose:** Stage 6 only creates a metadata file. Stage 7 physically reorganizes files by moving excluded clips to `quarantine/` subdirectory, leaving only the balanced dataset in the main directory.

---

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sample Rate | 16 kHz | Consistent audio processing |
| Clip Duration | 3.0 sec | Fixed-length segments for modeling |
| Audio Format | FLAC (Stage 3-4), WAV (Stage 5+) | Lossless compression |
| Channels | Mono | Single-channel audio |
| Min Separation | 1.5 sec | Between clips from same file |

## CSV Data Flow

```
Stage1 → id, en, rec, cnt, lat, lon, lic, q, length, smp, ...
  ↓
Stage2 → id, en, rec, cnt, lat, lon, lic, q, length, smp
  ↓
Stage4 → id, en, rec, cnt, lat, lon, lic, q, length, smp (filtered: no duplicates)
  ↓
Stage5 → xc_id, species, quality, recorder, country, latitude, longitude, license,
         source_file, source_duration_sec, original_length, sample_rate,
         clip_start_ms, clip_duration_sec, rms_energy, was_clipped, ...
  ↓
Stage6 → Same columns as Stage5 (balanced subset)
```

## Duplicate Detection Algorithm

Stage 4 uses acoustic similarity to detect duplicates:

1. **Embedding**: Computes normalized mel-spectrogram (128 mel bins, 8kHz max freq)
2. **Similarity**: Frame-wise cosine similarity with three metrics:
   - Mean similarity ≥ 0.997
   - Min similarity ≥ 0.985
   - 5th percentile ≥ 0.992
3. **Quarantine**: Perfect duplicates (≥ 0.999) - **newer recording** (higher XC ID) moved to `quarantine/`
4. **Output**: Near-duplicates saved to `duplicate_pairs.txt` for review

### Quarantine Structure
```
flacs/
├── quarantine/          # Newer duplicate files moved here
│   ├── Species_A/
│   └── Species_B/
├── Species_A/           # Older/unique files remain
└── Species_B/
```

## Output Files

### Stage-Specific Outputs
- **Stage 2**:
  - `Stage2_xc_successful_downloads.csv` - Successful downloads with full metadata
  - `Stage2_download_log.csv` - Complete download log
  - `Stage2_xc_failed_downloads.csv` - Failed downloads
- **Stage 4**:
  - `duplicate_pairs.txt` - Near-duplicate pairs
  - `Stage4_unique_flacs.csv` - **Non-quarantined files with metadata**
  - `quarantine/` - Quarantined duplicate files
- **Stage 5**: `clips_log.csv` - All extracted clips with enriched metadata
- **Stage 6**:
  - `balanced_clips.csv` - Metadata file listing clips in balanced dataset
  - `species_balance.png` - Before/after distribution visualization
- **Stage 7**: `quarantine/` - Excluded clips moved here (preserves data)

## Dependencies

```bash
pip install librosa numpy pandas requests tqdm soundfile matplotlib
```

## Usage Notes

### General Pipeline
1. **Complete workflow**: Stages 1-7 form a complete data curation pipeline
2. **Stage 5 + 6 + 7 workflow**: Run Stage 5 with `--no-quarantine`, then Stage 6 for balancing, then Stage 7 to move files
3. **Memory usage**: Stage 4 and 6 load data into RAM; consider batch processing for >100k files

### CLI Consistency
All stages now use standardized arguments:
- Input CSV: `--input-csv FILE`
- Output CSV: `--output-csv FILE`
- Input directory: `--inroot DIR`
- Output directory: `--outroot DIR`

### Stage-Specific
1. **Stage 2 - Skip short files**: Automatically skips files < 3 seconds
2. **Stage 4 - Metadata propagation**: Requires `--stage2-csv` to create enriched output CSV
3. **Stage 4 - Quarantine strategy**: Keeps older recordings (lower XC numbers)
4. **Stage 5 - Metadata enrichment**: Use `--metadata-csv Stage4_unique_flacs.csv` for full metadata
5. **Stage 5 - Balancing workflow**: Must use `--no-quarantine` if you plan to run Stage 6
6. **Stage 6 - Target size**: If not specified, defaults to 75% of total clips (rounded to nearest 1000)
7. **Stage 7 - Safety**: Moves files to quarantine instead of deleting (safer, preserves data)

## Recent Updates

**Metadata Enrichment & CLI Standardization (2025-12-23):**
1. **Stage 2**: Added metadata fields (recorder, country, lat/lon, license) to output CSV
2. **Stage 4**: Creates `Stage4_unique_flacs.csv` with metadata for non-quarantined files
3. **Stage 5**: Enriches output with metadata from Stage4 CSV (recorder, country, coordinates, license)
4. **CLI Standardization**: All stages now use consistent argument naming (--input-csv, --output-csv, --inroot, --outroot)
5. **Removed redundancy**: Removed `clip_start_sec` field (use `clip_start_ms` instead)
6. **Improved filtering**: Stage 2 skips files < 3 seconds and 4xx/5xx errors immediately

**Stage 6 & 7 Addition (2025-12-16):**
1. Added Stage 6: Species balancing with diversity optimization
   - XC ID diversity maximization (prioritizes unique recordings)
   - Quality-weighted selection (A > B > C > D ratings)
   - Gini coefficient tracking for distribution equality
   - Auto-calculated target size (75% of clips, configurable)
2. Added Stage 7: Move excluded files to quarantine
   - Physically reorganizes files after Stage 6 balancing
   - Safer than deletion (preserves excluded clips)
   - Verification of file counts

**Stage 4 Quarantine Strategy (2025-12-23):**
1. Changed quarantine logic to keep **older recordings** (lower XC numbers)
2. Creates `Stage4_unique_flacs.csv` for downstream processing

## Project Context

Part of the Malaysian Bird Audio Dataset (MyBAD) project for training bird sound classification models. The pipeline ensures high-quality, deduplicated training data with rich metadata from community-contributed recordings.

## Repository

https://github.com/mun3im/mybad/tree/main/positive_label_curation

## License

See main repository for license information.
