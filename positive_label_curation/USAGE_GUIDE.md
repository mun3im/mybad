# Southeast Asian Bird Dataset Pipeline - Usage Guide

## Overview

This pipeline fetches, processes, and balances Xeno-Canto bird recordings from Southeast Asian countries (Malaysia, Singapore, Indonesia, Brunei, Thailand) to create a high-quality, ecologically diverse dataset for CNN training.

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
  --out-csv xc_sea_birds.csv \
  --add-country-column

# Option 2: Fetch single country
python Stage1_xc_fetch_bird_metadata.py \
  --country Malaysia \
  --out-csv xc_my_birds.csv

# Option 3: Fetch specific subset of countries
python Stage1_xc_fetch_bird_metadata.py \
  --countries Malaysia Indonesia Thailand \
  --out-csv xc_subset.csv \
  --add-country-column
```

**Expected Output**:
- CSV file with metadata for all bird recordings
- Automatic deduplication by XC ID (removes recordings appearing in multiple countries)
- Country breakdown statistics

**Estimated Time**: 15-45 minutes depending on countries selected

---

### **Stage 2-4**: *(Not shown - previous stages in your pipeline)*

---

### **Stage 5: Extract 3-Second Clips**

**Script**: `Stage5_extract_3s_clips_from_flac.py`

**Purpose**: Extract high-quality 3-second WAV clips from FLAC files

**Key Features**:
- RMS-based clip selection (threshold: 0.001)
- Clipping detection and correction
- Dual workflow: RMS-only OR balanced (for Stage6)

**Usage** (Balanced Workflow - RECOMMENDED):

```bash
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /path/to/flac/files \
  --outroot /path/to/output/clips \
  --threshold 0.001 \
  --csv clips_log.csv \
  --no-quarantine
```

**Important**: Use `--no-quarantine` flag to keep ALL clips for Stage6 balancing!

**Output**:
- WAV clips (16kHz, PCM_16, 3 seconds)
- CSV log with RMS, timing, clipping status

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
  --csv clips_log.csv \
  --outroot /path/to/clips \
  --output balanced_clips.csv \
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
PRE-UNDERSAMPLING STATISTICS
====================================
Total species: 559
Total samples: 25,107
Average samples per species: 44.9
Gini coefficient: 0.531
Min samples (species): 1
Max samples (species): 420

POST-UNDERSAMPLING STATISTICS
====================================
Total species: 559
Total samples: 25,000
Average samples per species: 44.7
Gini coefficient: 0.449
Min samples (species): 1
Max samples (species): 160

DIVERSITY METRICS:
Unique XC recordings: 12,345
Samples per recording (avg): 2.02

Quality distribution:
  A: 8,450 (33.8%)
  B: 10,120 (40.5%)
  C: 5,230 (20.9%)
  D: 1,100 (4.4%)
  unknown: 100 (0.4%)

Gini coefficient improvement: 15.4%
```

**Visualization**: Side-by-side long-tail plots showing pre/post distribution

---

### **Stage 7: Move Excluded Files to Quarantine**

**Script**: `Stage7_move_to_quarantine.py`

**Purpose**: Physically reorganize files by moving clips excluded from balanced dataset to quarantine

**IMPORTANT**: Stage 6 only performs metadata balancing (creates CSV). Stage 7 performs file operations.

**Usage**:

```bash
python Stage7_move_to_quarantine.py \
  --csv balanced_clips.csv \
  --outroot /path/to/clips
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
Output directory: /Volumes/Evo/seabad-positive
Quarantine directory: /Volumes/Evo/seabad-positive/quarantine
============================================================

Loading balanced clips list...
Files to keep: 25,000

Scanning destination directory...
Total files in destination: 75,773
Files to move to quarantine: 50,773
Files to keep in destination: 25,000

Moving files to quarantine...
Moving files: 100%|██████████| 50773/50773 [00:04<00:00, 10667.29file/s]

Successfully moved 50,773 files to quarantine

Verifying...
Files remaining in destination: 25,000
Files in quarantine: 50,773
Expected in destination: 25,000

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
  --out-csv xc_sea_birds.csv \
  --add-country-column
```

### **Step 2-4**: *(Process and download as per your existing pipeline)*

### **Step 5: Extract Clips**
```bash
python Stage5_extract_3s_clips_from_flac.py \
  --inroot /Volumes/Evo/XC-All-SEA-Birds-16k-flac \
  --outroot /Volumes/Evo/XC-All-SEA-Birds-16k-wav-3s \
  --threshold 0.001 \
  --csv Stage5_sea_clips.csv \
  --no-quarantine
```

### **Step 6: Balance with Diversity**
```bash
python Stage6_balance_species.py \
  --csv Stage5_sea_clips.csv \
  --outroot /Volumes/Evo/XC-All-SEA-Birds-16k-wav-3s \
  --output balanced_sea_25k.csv \
  --target-size 25000 \
  --plots sea_species_balance.png
```

### **Step 7: Move Excluded Files to Quarantine**
```bash
python Stage7_move_to_quarantine.py \
  --csv balanced_sea_25k.csv \
  --outroot /Volumes/Evo/XC-All-SEA-Birds-16k-wav-3s
```

**What happens:**
- Moves 50,773 excluded files to `quarantine/` subdirectory
- Leaves 25,000 balanced files in main directory
- Safer than deletion - excluded files preserved for future use

---

## Expected Dataset Growth

### **Before (Malaysia only)**:
- Total recordings: ~5,000-8,000
- Final balanced dataset: ~16,000 clips (559 species)
- Avg samples/species: 28.6
- Gini: 0.449

### **After (5 countries)**:
- Total recordings: **~25,000-40,000** (estimated 3-5x increase)
- Final balanced dataset: **25,000+ clips** (700-900 species estimated)
- Avg samples/species: 28-35 (more balanced)
- Gini: Expected **<0.40** (better equality)

### **Geographic Coverage**:
- Peninsula Malaysia: ✅
- Borneo (Sarawak/Sabah): ✅✅ (Indonesia + Brunei overlap)
- Southern Thailand: ✅ (shares species with northern Malaysia)
- Singapore: ✅ (Sundaic lowland species)
- Indonesian Borneo: ✅ (massive Bornean endemic coverage)

---

## Quality Improvements from Stage6 Enhancements

### **Before Stage6 Enhancement**:
```
Selection: Pure RMS-based
Risk: Multiple clips from same recording
Example: Species with 100 samples from 10 recordings
  → Might pick 28 samples all from top 3 loudest recordings
```

### **After Stage6 Enhancement**:
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

### **Issue**: Stage6 fails with "KeyError: 'xc_id'"
**Solution**: Ensure Stage5 was run recently with the updated script (includes xc_id extraction)

### **Issue**: Very low Gini improvement in Stage6
**Diagnosis**: Dataset might already be well-balanced, or target_size too close to total samples
**Solution**: Increase target_size or check species distribution

### **Issue**: Duplicate XC IDs in Stage1 output
**Solution**: Script automatically deduplicates by ID (line 230). If still seeing duplicates, file a bug report.

---

## Performance Tips

1. **Parallel Processing**: Run Stage1 for different countries in parallel if you have multiple API keys
2. **Disk Space**: Estimate ~500MB per 1000 FLAC files, ~150MB per 1000 WAV clips
3. **API Rate Limiting**: Script includes 0.2s delay between requests (built-in)
4. **Memory Usage**: Stage6 loads entire CSV into memory. For >100k clips, consider chunking.

---

## Next Steps After Stage 7

Your balanced dataset is now ready for CNN training with:
- ✅ Ecological diversity (balanced species)
- ✅ Recording diversity (maximized unique XC IDs)
- ✅ Quality preference (A/B rated preferred)
- ✅ Geographic coverage (5 countries)
- ✅ Temporal diversity (1.5s min separation between clips)

**Recommended CNN Pipeline**:
1. Train/val/test split (80/10/10) stratified by species
2. Augmentation: SpecAugment, mixup, time stretching
3. Model: EfficientNet or ResNet on mel-spectrograms
4. Loss: Focal loss or label smoothing for long-tail handling

---

## Questions?

Check the individual script docstrings for detailed parameter descriptions:
```bash
python Stage1_xc_fetch_bird_metadata.py --help
python Stage5_extract_3s_clips_from_flac.py --help
python Stage6_balance_species.py --help
python Stage7_move_to_quarantine.py --help
```
