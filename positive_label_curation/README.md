# Positive Label Curation Pipeline

Audio data curation pipeline for the MyBAD (Malaysian Bird Audio Dataset) project. This pipeline fetches, downloads, processes, and curates bird audio recordings from Xeno-Canto.

## Pipeline Stages

### Stage 1: Fetch Metadata
**Script:** `Stage1_xc_fetch_bird_metadata.py`

Fetches bird species metadata from Xeno-Canto API for specified regions.

```bash
python Stage1_xc_fetch_bird_metadata.py --country MY --output xc_malaysia_birds.csv
```

### Stage 2: Download Audio
**Script:** `Stage2_xc_download_bird_mp3s.py`

Downloads MP3 audio files from Xeno-Canto based on metadata CSV.

```bash
python Stage2_xc_download_bird_mp3s.py xc_malaysia_birds.csv --output-dir downloads/
```

### Stage 3: Convert to FLAC
**Script:** `Stage3_convert_mp3_to_16k_flac.py`

Converts MP3 files to 16kHz mono FLAC format for consistent processing.

```bash
python Stage3_convert_mp3_to_16k_flac.py downloads/ --output-dir flac_16k/
```

### Stage 4: Detect Duplicates
**Script:** `Stage4_find_flac_duplicates.py`

Detects and quarantines duplicate audio clips using acoustic similarity.

```bash
python Stage4_find_flac_duplicates.py clips/ --recursive --output duplicate_pairs.txt
```

**Features:**
- Computes mel-spectrogram embeddings for each audio file
- Performs all-pairs similarity comparison using cosine similarity
- Automatically quarantines perfect duplicates (similarity ≥ 0.999999)
- Reports near-duplicates (similarity ≥ 0.997) for manual review

**See:** [Stage4_duplicate_detection_algorithm_documentation.md](./Stage4_Algo Document) for detailed algorithm documentation.

### Stage 5: Extract Clips
**Script:** `Stage5_extract_3s_clips_from_flac.py`

Extracts 3-second non-overlapping clips from full-length FLAC recordings.

```bash
python Stage5_extract_3s_clips_from_flac.py flac_16k/ --output-dir clips/ --duration 3.0
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Sample Rate | 16 kHz | Consistent audio processing |
| Clip Duration | 3.0 sec | Fixed-length segments for modeling |
| Audio Format | FLAC | Lossless compression |
| Channels | Mono | Single-channel audio |

## Duplicate Detection Algorithm

Stage 4 uses acoustic similarity to detect duplicates:

1. **Embedding**: Computes normalized mel-spectrogram (128 mel bins, 8kHz max freq)
2. **Similarity**: Frame-wise cosine similarity with three metrics:
   - Mean similarity ≥ 0.997
   - Min similarity ≥ 0.985
   - 5th percentile ≥ 0.992
3. **Quarantine**: Perfect duplicates (≥ 0.999999) moved to `quarantine/` folder
4. **Output**: Near-duplicates saved to `duplicate_pairs.txt` for review

### Quarantine Structure
```
clips/
├── quarantine/          # Duplicate files moved here
│   ├── Species_A/
│   └── Species_B/
├── Species_A/           # Clean files remain
└── Species_B/
```

## Output Files

- `duplicate_pairs.txt` - Tab-separated file with similarity scores and file pairs
- `quarantine/` - Folder containing quarantined duplicate files
- `Stage2_successful_downloads.csv` - Log of successfully downloaded files

## Dependencies

```bash
pip install librosa numpy pandas requests tqdm
```

## Usage Notes

1. **Recursive search**: Use `--recursive` flag for deeply nested directories
2. **Quarantine bypass**: Use `--no-quarantine` to only report duplicates without moving files
3. **Memory usage**: Stage 4 loads all embeddings into RAM; consider batch processing for >100k files
4. **Floating-point precision**: Perfect duplicate threshold is 0.999999 (not 1.0) to handle numerical precision

## Recent Fixes

**Stage 4 Quarantine Issues (2025-12-11):**
1. Fixed floating-point precision bug where scores of 0.999999... weren't detected as 1.0
2. Fixed quarantine logic to handle files appearing in multiple duplicate pairs
3. Changed threshold from 1.000 to 0.999999 to catch all perfect duplicates

## Project Context

Part of the Malaysian Bird Audio Dataset (MyBAD) project for training bird sound classification models. The pipeline ensures high-quality, deduplicated training data from community-contributed recordings.

## Repository

https://github.com/mun3im/mybad/tree/main/positive_label_curation

## License

See main repository for license information.
