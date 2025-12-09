# Stage4 Duplicate Detection Algorithm Documentation

## Overview

`Stage4_find_flac_duplicates.py` detects near-duplicate and perfect duplicate audio clips across subdirectories using acoustic similarity metrics. Perfect duplicates (similarity ≥ 0.999999) are automatically quarantined.

## Core Components

### 1. Audio Embedding (`AudioEmbedder`)

Converts audio files into normalized mel-spectrogram embeddings for comparison.

**Process:**
1. **Load audio**: Uses librosa with 16kHz sample rate, mono, kaiser_fast resampling
2. **Normalize length**: Pad or trim to exactly 3.0 seconds (48,000 samples)
3. **Compute mel-spectrogram**:
   - 128 mel bins
   - 512-sample FFT
   - 128-sample hop length
   - Max frequency: 8kHz
4. **Convert to log scale**: `librosa.power_to_db(mel, ref=np.max)`
5. **L2 normalize frames**: Per-frame normalization for cosine similarity

**Output:** `(128, n_frames)` array of normalized log-mel features

### 2. Similarity Calculation (`SimilarityCalculator`)

Computes frame-wise cosine similarity between two embeddings.

**Metrics:**
- **Mean similarity**: Average cosine similarity across all frames
- **Min similarity**: Minimum cosine similarity across all frames
- **5th percentile similarity**: 5th percentile of frame similarities (robust to outliers)

**Formula:**
```python
sims = np.sum(emb1[:, :min_frames] * emb2[:, :min_frames], axis=0)
# Clipped to [-1.0, 1.0] range
```

**Thresholds for "similar" classification:**
- Mean similarity ≥ 0.997
- Min similarity ≥ 0.985
- 5th percentile similarity ≥ 0.992

**Perfect duplicate threshold:** ≥ 0.999999 (accounts for floating-point precision)

### 3. Duplicate Detection (`DuplicateFinder`)

Performs all-pairs comparison to find duplicates.

**Process:**
1. Compute embeddings for all valid audio files
2. Compare each pair (i, j) where i < j
3. Categorize pairs:
   - **Perfect duplicates**: mean_sim ≥ 0.999999
   - **Near-duplicates**: Meets all three similarity thresholds

**Complexity:** O(n²) where n is the number of files

### 4. Quarantine Logic (`QuarantineManager`)

Moves perfect duplicates to a quarantine folder while preserving directory structure.

**Strategy:**
- For each perfect duplicate pair (file1, file2):
  1. **Prefer file2**: Move file2 if not already quarantined
  2. **Fallback to file1**: Move file1 if file2 was already quarantined
  3. **Skip**: Only if both files already quarantined

This ensures every perfect duplicate pair has at least one file removed.

**Quarantine structure:**
```
project_root/
├── quarantine/
│   ├── Species_A/
│   │   └── xc12345_A.flac
│   └── Species_B/
│       └── xc67890_B.flac
├── Species_A/
│   └── xc12346_B.flac  (kept)
└── Species_B/
    └── xc67891_A.flac  (kept)
```

## Algorithm Flow

```
1. Collect audio files (.wav, .flac)
   ↓
2. Compute embeddings for all files
   ↓
3. All-pairs similarity comparison
   ↓
4. Categorize pairs:
   - Perfect duplicates (≥ 0.999999)
   - Near-duplicates (≥ thresholds)
   ↓
5. Quarantine perfect duplicates
   ↓
6. Write near-duplicates to output file
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `TARGET_SR` | 16000 Hz | Sample rate for audio loading |
| `TARGET_DURATION` | 3.0 sec | Fixed clip duration |
| `N_MELS` | 128 | Number of mel frequency bins |
| `HOP_LENGTH` | 128 samples | Hop length for STFT |
| `MIN_DURATION` | 3.0 sec | Minimum duration filter |
| `MEAN_SIM_THRESHOLD` | 0.997 | Mean similarity for near-duplicates |
| `MIN_SIM_THRESHOLD` | 0.985 | Min similarity for near-duplicates |
| `P5_SIM_THRESHOLD` | 0.992 | 5th percentile for near-duplicates |
| `PERFECT_DUPLICATE_THRESHOLD` | 0.999999 | Perfect duplicate threshold |

## Floating-Point Precision

The perfect duplicate threshold is set to `0.999999` instead of `1.0` to account for floating-point arithmetic precision. Scores displayed as `1.000000` may internally be `0.9999999...` due to numerical computation.

## Output Format

**Near-duplicates file** (`duplicate_pairs.txt`):
```
# 75 perfect duplicate pair(s) moved to quarantine/

1.000000    Species_A/xc123_A.flac    Species_A/xc456_B.flac
0.999994    Species_B/xc789_C.flac    Species_B/xc012_A.flac
0.998500    Species_C/xc345_B.flac    Species_D/xc678_A.flac
```

Format: `similarity_score<TAB>file1_path<TAB>file2_path`

## Usage

**Basic usage:**
```bash
python Stage4_find_flac_duplicates.py /path/to/audio/directory
```

**Options:**
- `--recursive` / `-r`: Search all subdirectories (not just one level deep)
- `--output` / `-o`: Specify output file (default: `duplicate_pairs.txt`)
- `--no-quarantine`: Skip quarantine step, only report duplicates

**Example:**
```bash
python Stage4_find_flac_duplicates.py ./bird_clips --recursive --output duplicates.txt
```

## Implementation Notes

1. **Memory usage**: Stores all embeddings in memory; may need optimization for very large datasets (>100k files)

2. **Quarantine safety**: Uses `shutil.move()` which preserves file atomicity. Subfolder structure is maintained in quarantine.

3. **Error handling**: Skips files that fail to load and logs errors to stderr

4. **Progress tracking**: Uses `tqdm` for progress bars during embedding and comparison phases

## Known Limitations

1. **Computational complexity**: O(n²) comparison may be slow for large datasets
2. **Fixed duration**: Only works with 3-second clips
3. **No transposition detection**: Does not detect pitch-shifted duplicates
4. **No time-stretch detection**: Does not detect time-stretched duplicates
5. **Memory bound**: Loads all embeddings into RAM

## Future Improvements

1. Implement approximate nearest neighbor search (e.g., FAISS) for O(n log n) complexity
2. Add support for variable-length audio
3. Implement pitch-invariant and tempo-invariant similarity metrics
4. Add parallel processing for embedding computation
5. Implement streaming/batch processing for very large datasets
