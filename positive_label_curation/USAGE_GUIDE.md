# Southeast Asian Bird Dataset Pipeline - Complete Usage Guide

## Overview

This pipeline fetches, processes, and balances Xeno-Canto bird recordings from Southeast Asian countries (Malaysia, Singapore, Indonesia, Brunei, Thailand) to create a high-quality, ecologically diverse dataset with rich metadata for CNN training.

**Key Innovation**: Metadata enrichment throughout the pipeline - from download to final clips, preserving recorder, country, coordinates, and license information.

---

## Geographic Expansion Rationale

**Original**: Malaysia only
**Expanded**: Malaysia + Singapore + Indonesia + Brunei + Thailand

**Why?** All countries share borders with Malaysia:
- **Singapore**: Southern border (Johor causeway)
- **Indonesia**: Borneo border (Kalimantan/Sarawak/Sabah) + maritime borders
- **Brunei**: Embedded within Sarawak, Borneo
- **Thailand**: Northern border (Perlis/Kedah/Perak)

**Benefit**: Shared ecosystems, migratory species, and similar avifauna → significantly increases dataset size while maintaining ecological relevance.

---

## Pipeline Stages

### **Stage 1: Fetch Xeno-Canto Metadata**

**Script**: `Stage1_xc_fetch_bird_metadata.py`

**Purpose**: Download bird recording metadata from Xeno-Canto API

**Prerequisites**:
```bash
export XENO_API_KEY="your_xeno_canto_api_key_here"
```

**Usage Options**:

```bash
# Option 1: Fetch ALL Southeast Asian countries (RECOMMENDED for max dataset size)
python Stage1_xc_fetch_bird_metadata.py \
  --country all \
  --output-csv xc_sea_birds.csv \
  --add-country-column

# Option 2: Fetch single country
python Stage1_xc_fetch_bird_metadata.py \
  --country Malaysia \
  --output-csv xc_my_birds.csv

# Option 3: Fetch specific subset of countries
python Stage1_xc_fetch_bird_metadata.py \
  --countries Malaysia Indonesia Thailand \
  --output-csv xc_subset.csv \
  --add-country-column
```

**Expected Output**:
- CSV file with metadata for all bird recordings
- Automatic deduplication by XC ID (removes recordings appearing in multiple countries)
- Country breakdown statistics

**Output Columns**:
```
id, gen, sp, ssp, grp, en, rec, cnt, loc, lat, lon, alt, type, sex, stage,
method, url, file, file-name, lic, q, length, time, date, uploaded, also,
rmk, animal-seen, playback-used, temp, regnr, auto, dvc, mic, smp, fetch_country
```

**Estimated Time**: 15-45 minutes depending on countries selected

---

### **Stage 2: Download Audio Files** ⭐ (Metadata Creation)

**Script**: `Stage2_xc_download_bird_mp3s.py`

**Purpose**: Download MP3 files and create metadata CSV for downstream processing

**Key Features**:
- **Metadata extraction**: Creates CSV with recorder, country, coordinates, license
- **Smart filtering**: Skips files < 3 seconds automatically
- **Error handling**: Skips 4xx/5xx errors immediately (no retries)
- **Multiple outputs**: Download log, success CSV, failed CSV

**Usage**:

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

# Dry run (no actual downloads)
python Stage2_xc_download_bird_mp3s.py \
  --input-csv Stage1_xc_sea_birds.csv \
  --outroot /path/to/output \
  --dry-run
```

**Output Files** (saved in script directory):
- `Stage2_xc_successful_downloads.csv` - **Critical for downstream stages**
  - Columns: `id, en, rec, cnt, lat, lon, lic, q, length, smp`
- `Stage2_download_log.csv` - Complete log with status for every row
- `Stage2_xc_failed_downloads.csv` - Failed downloads for manual retry

**File Organization**:
```
/Volumes/Evo/xc-asean-mp3s/
├── Species_Name_1/
│   ├── xc123456_A.mp3
│   ├── xc123457_B.mp3
│   └── ...
├── Species_Name_2/
│   └── ...
```

---

### **Stage 3: Convert to 16kHz FLAC**

**Script**: `Stage3_convert_mp3_to_16k_flac.py`

**Purpose**: Convert MP3 files to consistent 16kHz mono FLAC format

**Usage**:

```bash
# Basic conversion
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /Volumes/Evo/xc-asean-mp3s \
  --outroot /Volumes/Evo/xc-asean-flacs

# With parallel workers
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /path/to/mp3s \
  --outroot /path/to/flacs \
  --workers 8

# With normalization
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /path/to/mp3s \
  --outroot /path/to/flacs \
  --normalize-db -1.0 \
  --allow-boost
```

**Output**:
- FLAC files (16kHz, mono): `{species}/xc{id}_{quality}.flac`
- Conversion log CSV

---

### **Stage 4: Detect and Remove Duplicates** ⭐ (Metadata Propagation)

**Script**: `Stage4_find_flac_duplicates.py`

**Purpose**: Find duplicates and create metadata CSV of unique files

**Key Features**:
- **Acoustic similarity** detection using mel-spectrograms
- **Keeps older recordings** (lower XC numbers) when duplicates found
- **Creates Stage4_unique_flacs.csv** - Filtered metadata for downstream

**Usage**:

```bash
# Find duplicates and create unique files CSV
python Stage4_find_flac_duplicates.py /Volumes/Evo/xc-asean-flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --stage4-csv Stage4_unique_flacs.csv

# Dry run to preview
python Stage4_find_flac_duplicates.py /path/to/flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --dry-run

# Search recursively
python Stage4_find_flac_duplicates.py /path/to/flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --recursive
```

**Output**:
- `duplicate_pairs.txt` - Near-duplicates for manual review
- `quarantine/` folder - Newer duplicate files moved here
- **`Stage4_unique_flacs.csv`** - Metadata for non-quarantined files (same columns as Stage2)

**Quarantine Strategy**:
- For duplicate pair: Keeps **older** recording (lower XC ID)
- Quarantines **newer** recording (higher XC ID)
- Preserves original metadata in output CSV

---

### **Stage 5: Extract 3-Second Clips** ⭐ (Metadata Enrichment)

**Script**: `Stage5_extract_3s_clips_from_flac.py`

**Purpose**: Extract high-quality 3-second WAV clips with **enriched metadata**

**Key Features**:
- RMS-based clip selection (threshold: 0.001)
- Clipping detection and soft limiting
- **Metadata enrichment** from Stage4 CSV
- Dual workflow: RMS-only OR balanced (for Stage6)

**Usage** (Balanced Workflow - RECOMMENDED):

```bash
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /Volumes/Evo/xc-asean-flacs \
  --outroot /Volumes/Evo/xc-asean-clips \
  --output-csv Stage5_clips_log.csv \
  --metadata-csv Stage4_unique_flacs.csv \
  --threshold 0.001 \
  --no-quarantine
```

**Usage** (RMS-only Workflow - immediate dataset):

```bash
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /Volumes/Evo/xc-asean-flacs \
  --outroot /Volumes/Evo/xc-asean-clips \
  --output-csv Stage5_clips_log.csv \
  --metadata-csv Stage4_unique_flacs.csv \
  --max-clips 25000
```

**Important**:
- Use `--no-quarantine` flag to keep ALL clips for Stage6 balancing!
- Use `--metadata-csv` to enrich output with recorder, country, coordinates, license

**Output**:
- WAV clips (16kHz, PCM_16, 3 seconds): `xc{id}_{quality}_{start_ms}.wav`
- **`Stage5_clips_log.csv`** with enriched metadata:

**Output CSV Columns**:
```
xc_id, species, quality,
recorder, country, latitude, longitude, license,        # From Stage4 metadata
source_file, source_duration_sec, original_length, sample_rate,
clip_start_ms, clip_duration_sec, rms_energy, was_clipped,
out_filename, out_path
```

**Metadata Fields Explained**:
- **recorder**: Person who recorded the audio (from XC metadata)
- **country**: Country where recorded
- **latitude, longitude**: GPS coordinates
- **license**: Creative Commons license
- **original_length**: Original file length in MM:SS format (from XC)
- **source_duration_sec**: Actual FLAC file duration in seconds
- **sample_rate**: Original sample rate from XC metadata

---

### **Stage 6: Species Balancing with Diversity Optimization** ⭐

**Script**: `Stage6_balance_species.py`

**Purpose**: Balance species representation while maximizing recording diversity

**NEW Features** (Enhanced):
1. ✅ **XC ID Diversity**: Prioritizes samples from different recordings
2. ✅ **Quality Weighting**: Prefers A-rated > B > C > D recordings
3. ✅ **Progress Feedback**: Real-time tqdm progress bars
4. ✅ **Diversity Metrics**: Reports unique XC IDs and quality distribution

**Usage**:

```bash
python Stage6_balance_species.py \
  --input-csv Stage5_clips_log.csv \
  --outroot /Volumes/Evo/xc-asean-clips \
  --output-csv balanced_clips.csv \
  --target-size 25000 \
  --plots species_balance.png
```

**Selection Strategy**:

1. **Per-species cap calculation**: `target_size / num_species`
2. **Diversity-first sampling**:
   - Priority 1: Unique XC recordings (one sample per recording first)
   - Priority 2: Quality rating (A > B > C > D)
   - Priority 3: RMS energy (within quality tier)
3. **Backfill optimization**: If under target, adds more samples prioritizing:
   - New XC IDs not yet in dataset (+100 bonus)
   - Higher quality ratings
   - Higher RMS energy

**Expected Output**:

```
============================================================
Stage 6: Species-Level Undersampling for Ecological Diversity
============================================================
Input CSV: Stage5_results.csv
Output CSV: balanced_clips.csv
Target dataset size: 20,000
============================================================
Loaded 38,553 valid clips from CSV
Note: Ensure Stage5 was run with --no-quarantine flag for balanced workflow
============================================================
PRE-UNDERSAMPLING STATISTICS
============================================================
Total species: 1676
Total samples: 38,553
Average samples per species: 23.0
Gini coefficient: 0.606
Min samples (species): 1
Max samples (species): 786
Top 5 species by sample count:
  Identity_unknown: 786 samples
  Greater_Racket-tailed_Drongo: 332 samples
  Olive-backed_Sunbird: 261 samples
  Blue-eared_Barbet: 220 samples
  Mountain_Tailorbird: 214 samples
Quality grade distribution (all clips):
  A: 11,213 (29.1%)
  B: 18,428 (47.8%)
  C: 7,094 (18.4%)
  D: 1,274 (3.3%)
  U (unknown): 544 (1.4%)
Quality distribution per species (summary):
  Species with grade A clips: 1262/1676
  Species with grade B clips: 1496/1676
  Species with grade C clips: 1231/1676
  Species with grade D clips: 538/1676
  Species with grade U clips: 267/1676
============================================================
Applying species-level undersampling...
Applying diversity-aware undersampling (base cap: 11 samples/species)...
Processing species: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1676/1676 [00:01<00:00, 1312.42species/s]
Initial balance yielded 13,065 samples. Need 6,935 more to reach target.
Building backfill pool: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1676/1676 [00:01<00:00, 1549.26species/s]
Backfilling with 6,935 samples (prioritizing new XC IDs)...
Backfilling: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6935/6935 [00:06<00:00, 1067.35sample/s]
Added 6,935 more samples to reach closer to target
============================================================
POST-UNDERSAMPLING STATISTICS
============================================================
Total species: 1676
Total samples: 20,000
Average samples per species: 11.9
Gini coefficient: 0.475
Min samples (species): 1
Max samples (species): 250
Top 5 species by sample count:
  Identity_unknown: 250 samples
  Greater_Racket-tailed_Drongo: 118 samples
  White-chested_Babbler: 107 samples
  Rufous-tailed_Tailorbird: 103 samples
  Pygmy_Cupwing: 90 samples
DIVERSITY METRICS:
Unique XC recordings: 20,000
Samples per recording (avg): 1.00
Quality distribution:
  A: 11,213 (56.1%)
  B: 7,326 (36.6%)
  C: 1,171 (5.9%)
  D: 202 (1.0%)
  U (unknown): 88 (0.4%)
============================================================
Gini coefficient improvement: 21.6%
Samples removed: 18,553
Balanced dataset saved to: balanced_clips.csv
Generating distribution histograms...
Long-tail distribution plot saved to: species_balance.png
============================================================
Species balancing complete!
============================================================
```

**Visualization**: Side-by-side long-tail plots showing pre/post distribution

**Output**:
- `balanced_clips.csv` - Same columns as Stage5 (metadata preserved)
- `species_balance.png` - Distribution visualization

---

### **Stage 7: Move Excluded Files to Quarantine**

**Script**: `Stage7_move_to_quarantine.py`

**Purpose**: Physically reorganize files by moving clips excluded from balanced dataset to quarantine

**IMPORTANT**: Stage 6 only performs metadata balancing (creates CSV). Stage 7 performs file operations.

**Usage**:

```bash
python Stage7_move_to_quarantine.py \
  --input-csv balanced_clips.csv \
  --outroot /Volumes/Evo/xc-asean-clips

# Preview what would be moved
python Stage7_move_to_quarantine.py \
  --input-csv balanced_clips.csv \
  --outroot /path/to/clips \
  --dry-run
```

**What it does**:

1. Reads `balanced_clips.csv` to identify files to KEEP
2. Scans output directory for all audio files
3. Moves files NOT in the balanced CSV to `quarantine/` subdirectory
4. Verifies final file counts match expectations

**Example Output**:

```
============================================================
Stage 7: Move Excluded Files to Quarantine
============================================================
Balanced CSV: balanced_clips.csv
Output directory: /Volumes/Evo/xc-asean-3s-wav
Quarantine directory: /Volumes/Evo/xc-asean-3s-wav/quarantine
============================================================

Loading balanced clips list...
Files to keep: 20,000

Scanning destination directory...
Total files in destination: 38,553
Files to move to quarantine: 18,553
Files to keep in destination: 20,000

Moving files to quarantine...
Moving files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18553/18553 [00:02<00:00, 6651.66file/s]

Successfully moved 18,553 files to quarantine

Verifying...
Files remaining in destination: 20,000
Files in quarantine: 18,553
Expected in destination: 20,000

✓ Success! File counts match.
```

**Why this stage exists:**
- Stage 6 is non-destructive (metadata only) for safety
- Stage 7 gives you control over when files are moved
- Quarantine preserves excluded files instead of deleting them
- Allows reviewing balanced dataset before committing to file operations

---

## Recommended Full Workflow

### **Step 1: Fetch Metadata (Expanded)**
```bash
export XENO_API_KEY="your_key"

python Stage1_xc_fetch_bird_metadata.py \
  --country all \
  --output-csv Stage1_xc_sea_birds.csv \
  --add-country-column
```

### **Step 2: Download MP3s (Creates Metadata CSV)**
```bash
python Stage2_xc_download_bird_mp3s.py \
  --input-csv Stage1_xc_sea_birds.csv \
  --outroot /Volumes/Evo/xc-asean-mp3s
```

**Output**: `Stage2_xc_successful_downloads.csv` (critical for later stages)

### **Step 3: Convert to FLAC**
```bash
python Stage3_convert_mp3_to_16k_flac.py \
  --inroot /Volumes/Evo/xc-asean-mp3s \
  --outroot /Volumes/Evo/xc-asean-flacs \
  --workers 8
```

### **Step 4: Remove Duplicates (Creates Unique Metadata CSV)**
```bash
python Stage4_find_flac_duplicates.py /Volumes/Evo/xc-asean-flacs \
  --stage2-csv Stage2_xc_successful_downloads.csv \
  --stage4-csv Stage4_unique_flacs.csv
```

**Output**: `Stage4_unique_flacs.csv` (metadata for non-duplicate files)

### **Step 5: Extract Clips (With Metadata Enrichment)**
```bash
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /Volumes/Evo/xc-asean-flacs \
  --outroot /Volumes/Evo/xc-asean-clips \
  --output-csv Stage5_clips_log.csv \
  --metadata-csv Stage4_unique_flacs.csv \
  --threshold 0.001 \
  --no-quarantine
```

**Output**: `Stage5_clips_log.csv` (enriched with recorder, country, coordinates, license)

### **Step 6: Balance with Diversity**
```bash
python Stage6_balance_species.py \
  --input-csv Stage5_clips_log.csv \
  --outroot /Volumes/Evo/xc-asean-clips \
  --output-csv balanced_clips.csv \
  --target-size 25000 \
  --plots species_balance.png
```

**Output**: `balanced_clips.csv` (metadata only, preserves all Stage5 columns)

### **Step 7: Move Excluded Files to Quarantine**
```bash
python Stage7_move_to_quarantine.py \
  --input-csv balanced_clips.csv \
  --outroot /Volumes/Evo/xc-asean-clips
```

**What happens:**
- Moves 13,426 excluded files to `quarantine/` subdirectory
- Leaves 25,000 balanced files in main directory
- Safer than deletion - excluded files preserved for future use

---

## Metadata Flow Through Pipeline

```
Stage1 (XC API) → Full XC metadata (50+ columns)
  ↓
Stage2 (Download) → Filtered metadata CSV
  id, en, rec, cnt, lat, lon, lic, q, length, smp
  ↓
Stage4 (Deduplicate) → Same columns, filtered rows (no duplicates)
  id, en, rec, cnt, lat, lon, lic, q, length, smp
  ↓
Stage5 (Extract Clips) → Enriched clip metadata
  xc_id, species, quality,
  recorder, country, latitude, longitude, license,
  source_file, source_duration_sec, original_length, sample_rate,
  clip_start_ms, clip_duration_sec, rms_energy, was_clipped,
  out_filename, out_path
  ↓
Stage6 (Balance) → Same columns as Stage5, balanced subset
  ↓
Final Dataset → 25,000 clips with full provenance metadata
```

---

## Expected Dataset Growth

### **Before (Malaysia only)**:
- Total recordings: ~5,000-8,000
- Final balanced dataset: ~16,000 clips (559 species)
- Avg samples/species: 28.6
- Gini: 0.449

### **After (5 countries)**:
- Total recordings: **~25,000-40,000** (estimated 3-5x increase)
- After Stage5: **~38,000+ clips**
- Final balanced dataset: **25,000 clips** (700-900 species estimated)
- Avg samples/species: 28-35 (more balanced)
- Gini: Expected **<0.40** (better equality)

### **Geographic Coverage**:
- Peninsula Malaysia: ✅
- Borneo (Sarawak/Sabah): ✅✅ (Indonesia + Brunei overlap)
- Southern Thailand: ✅ (shares species with northern Malaysia)
- Singapore: ✅ (Sundaic lowland species)
- Indonesian Borneo: ✅ (massive Bornean endemic coverage)

---

## Quality Improvements from Enhancements

### **Stage2 Metadata Creation**:
**Before**: Only filenames tracked
**After**: Full metadata (recorder, country, coordinates, license) preserved through pipeline

### **Stage4 Duplicate Strategy**:
**Before**: Random quarantine
**After**: Strategic - keeps **older** recordings (lower XC IDs)

### **Stage5 Metadata Enrichment**:
**Before**: Only clip-level data
**After**: Enriched with recorder, country, coordinates, license from upstream

### **Stage6 Diversity Optimization**:
**Before**:
```
Selection: Pure RMS-based
Risk: Multiple clips from same recording
Example: Species with 100 samples from 10 recordings
  → Might pick 28 samples all from top 3 loudest recordings
```

**After**:
```
Selection: Diversity-optimized (XC ID + Quality + RMS)
Guarantee: Maximum unique recordings first
Example: Species with 100 samples from 20 recordings
  → Picks best sample from each of 20 recordings (20 samples)
  → Then adds 8 more high-quality/loud clips (28 total)
  → Result: 20 unique recordings vs 3 (6.7x diversity!)
```

---

## File Naming Convention

Xeno-Canto files follow this pattern:
```
xc{id}_{quality}.{ext}
```

Examples:
- `xc123456_A.flac` → XC ID: 123456, Quality: A (best)
- `xc789012_C.flac` → XC ID: 789012, Quality: C (acceptable)

Extracted clips:
```
xc{id}_{quality}_{start_ms}.wav
```

Examples:
- `xc123456_A_4700.wav` → From xc123456_A.flac, starting at 4.7 seconds
- `xc789012_C_12300.wav` → From xc789012_C.flac, starting at 12.3 seconds

---

## Troubleshooting

### **Issue**: "No XENO_API_KEY found"
**Solution**:
```bash
export XENO_API_KEY="your_api_key"
# Or add to ~/.bashrc or ~/.zshrc for persistence
```

### **Issue**: Stage5 missing metadata fields in output CSV
**Solution**: Ensure you used `--metadata-csv Stage4_unique_flacs.csv` flag

### **Issue**: Stage6 fails with "KeyError: 'xc_id'"
**Solution**: Ensure Stage5 was run recently with the updated script (includes xc_id extraction)

### **Issue**: Very low Gini improvement in Stage6
**Diagnosis**: Dataset might already be well-balanced, or target_size too close to total samples
**Solution**: Increase target_size or check species distribution

### **Issue**: Stage4 doesn't create Stage4_unique_flacs.csv
**Solution**: Ensure you provided `--stage2-csv Stage2_xc_successful_downloads.csv` argument

---

## Performance Tips

1. **Parallel Processing**: Run Stage1 for different countries in parallel if you have multiple API keys
2. **Disk Space**: Estimate ~500MB per 1000 FLAC files, ~150MB per 1000 WAV clips
3. **API Rate Limiting**: Script includes 0.2s delay between requests (built-in)
4. **Memory Usage**: Stage6 loads entire CSV into memory. For >100k clips, consider chunking.

---

## CLI Argument Standardization

All stages now use consistent argument naming:

| Argument Type | Standard Name | Metavar | Example |
|--------------|---------------|---------|---------|
| Input CSV | `--input-csv` | FILE | `--input-csv Stage1_data.csv` |
| Output CSV | `--output-csv` | FILE | `--output-csv results.csv` |
| Input Directory | `--inroot` | DIR | `--inroot /path/to/input` |
| Output Directory | `--outroot` | DIR | `--outroot /path/to/output` |
| Dry Run | `--dry-run` | (flag) | `--dry-run` |

**Use `--help` for any stage to see all options:**
```bash
python Stage1_xc_fetch_bird_metadata.py --help
python Stage2_xc_download_bird_mp3s.py --help
python Stage5_extract_3s_clips_from_flac.py --help
```

---

## Next Steps After Stage 7

Your balanced dataset is now ready for CNN training with:
- ✅ Ecological diversity (balanced species)
- ✅ Recording diversity (maximized unique XC IDs)
- ✅ Quality preference (A/B rated preferred)
- ✅ Geographic coverage (5 countries)
- ✅ Temporal diversity (1.5s min separation between clips)
- ✅ **Full metadata** (recorder, country, coordinates, license)

**Recommended CNN Pipeline**:
1. Train/val/test split (80/10/10) stratified by species
2. Augmentation: SpecAugment, mixup, time stretching
3. Model: EfficientNet or ResNet on mel-spectrograms
4. Loss: Focal loss or label smoothing for long-tail handling
5. **Attribution**: Use recorder metadata for proper attribution in publications

---

## Metadata Usage Examples

With the enriched metadata, you can now:

1. **Filter by quality**:
   ```python
   df = pd.read_csv('balanced_clips.csv')
   high_quality = df[df['quality'].isin(['A', 'B'])]
   ```

2. **Geographic analysis**:
   ```python
   df['lat'] = pd.to_numeric(df['latitude'])
   df['lon'] = pd.to_numeric(df['longitude'])
   # Plot geographic distribution
   ```

3. **License compliance**:
   ```python
   license_dist = df['license'].value_counts()
   # Ensure proper attribution per license
   ```

4. **Recorder attribution**:
   ```python
   top_contributors = df['recorder'].value_counts().head(10)
   # Acknowledge top contributors in publications
   ```

---

## Questions?

Check the individual script docstrings for detailed parameter descriptions:
```bash
python Stage1_xc_fetch_bird_metadata.py --help
python Stage2_xc_download_bird_mp3s.py --help
python Stage3_convert_mp3_to_16k_flac.py --help
python Stage4_find_flac_duplicates.py --help
python Stage5_extract_3s_clips_from_flac.py --help
python Stage6_balance_species.py --help
python Stage7_move_to_quarantine.py --help
```

Or consult the [README.md](README.md) for quick reference.
