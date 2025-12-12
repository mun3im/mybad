# Negative Label Curation Pipeline

Automated pipeline for extracting 25,000 high-quality negative (non-bird) audio samples for bird detection model training.

## Overview

This pipeline curates negative samples from multiple acoustic datasets to train robust bird detection models for field deployment. All samples are 3-second mono clips at 16kHz with consistent quality filtering across stages.

**Target:** 25,000 negative samples
**Output:** Acoustically diverse non-bird sounds relevant to real-world deployment

## Why These Datasets?

We carefully selected four complementary datasets:

| Dataset | Contribution | Why Chosen |
|---------|-------------|------------|
| **DCASE** (BirdVox, Freefield1010, Warblrb10k) | ~21,000 samples | Expert-annotated environmental soundscapes (wind, rain, insects, ambient noise) from bird detection challenges |
| **ESC-50** | ~350 samples | Diverse environmental sounds (50 categories) including edge cases like footsteps, water, mechanical sounds |
| **FSC-22** | Variable | Other biological sounds (mammals, amphibians, invertebrates) that co-occur with birds |
| **DATASEC** | Dynamic (fills to 25k) | Environmental sounds for realistic  field environments |

## Pipeline Stages

### Stage 1: DCASE Datasets
```bash
python Stage1_extract_dcase.py
```

Processes three DCASE bird detection datasets:
- Filters for `hasbird == 0` annotations
- Extracts center 3s clips from audio files
- Parallel processing with quality filtering
- **Output:** `/Volumes/Evo/mybad2/negative/{bv,ff,wb}/`

### Stage 2: ESC-50
```bash
python Stage2_extract_esc50.py
```

Extracts non-bird environmental sounds:
- Excludes bird categories: `chirping_birds`, `crow`, `rooster`, `hen`
- All ESC-50 files are exactly 5s → extracts center 3s
- **Output:** `/Volumes/Evo/mybad2/negative/esc/`

### Stage 3: FSC-22
```bash
python Stage3_extract_fsc22.py
```

Extracts non-bird biological sounds:
- Excludes classes 23 (BirdChirping) and 24 (WingFlapping)
- Processes few-shot bioacoustic event detection data
- **Output:** `/Volumes/Evo/mybad2/negative/fsc/`

### Stage 4: DATASEC (Dynamic Allocation)
```bash
python Stage4_extract_datasec.py
```

Intelligently fills remaining quota to reach 25,000 samples:

**Phase 1: Quality Filtering**
- RMS energy ≥ 0.0001
- Peak amplitude ≤ 0.98 (no clipping)
- Dynamic range ≥ 0.1

**Phase 2: Deployment-Aware Allocation**
- 5% from music folders (unlikely in jungle/field deployment)
- 95% from voice/speech folders (researchers, tourists, rangers)

**Phase 3: Diversity Maximization**
- Computes spectral diversity score for each clip
- Selects top-N most diverse samples per category
- **Output:** `/Volumes/Evo/mybad2/negative/datasec/`

## Quick Start

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### Run with Options
```bash
# Skip stages (uses existing files)
python run_pipeline.py --skip-stage1 --skip-stage2

# Adjust DATASEC music ratio (default: 5%)
python run_pipeline.py --music-ratio 0.1
```

### Run Individual Stages
```bash
python Stage1_extract_dcase.py  # DCASE datasets
python Stage2_extract_esc50.py   # ESC-50
python Stage3_extract_fsc22.py   # FSC-22
python Stage4_extract_datasec.py # DATASEC (dynamic)
```

## Quality Filtering

All stages apply consistent quality thresholds:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| **Min Duration** | 3.0s | Ensure sufficient audio |
| **RMS Energy** | ≥ 0.0001 | Remove silence/noise floor |
| **Zero Check** | All samples | Remove corrupted audio |
| **Peak Amplitude** | ≤ 0.98 (Stage 4) | Detect clipping |
| **Dynamic Range** | ≥ 0.1 (Stage 4) | Ensure variation |

## Algorithm: Spectral Diversity Score

For large folders (music, voices), we maximize diversity by scoring each clip:

```
diversity_score = spectral_centroid/8000 +
                  spectral_rolloff/8000 +
                  spectral_bandwidth/4000 +
                  10 × zero_crossing_rate
```

Clips are ranked by score and top-N most diverse samples are selected.

## Output Structure

```
/Volumes/Evo/mybad2/negative/
├── bv/          # BirdVox-DCASE-20k negatives
├── ff/          # Freefield1010 negatives
├── wb/          # Warblrb10k negatives
├── esc/         # ESC-50 negatives
├── fsc/         # FSC-22 negatives
└── datasec/     # DATASEC negatives
```

All files: **3 seconds, 16kHz, mono, .wav format**

## Logging & Reproducibility

Each stage produces detailed logs:

- `Stage1_rejections.txt` - DCASE processing
- `Stage2_rejections.txt` - ESC-50 processing
- `Stage3_rejections.txt` - FSC-22 processing
- `Stage4_rejections.txt` - DATASEC processing
- `pipeline_orchestrator.log` - Overall execution

Logs include:
- Every skipped file with reason (too short, too quiet, missing, corrupted)
- Processing errors and exceptions
- Summary statistics per stage

## Dependencies

```bash
pip install pandas numpy librosa soundfile pydub tqdm
```

**Requirements:**
- Python 3.7+
- librosa (audio loading/processing)
- soundfile (audio writing)
- pydub (audio conversion)
- pandas (metadata handling)
- tqdm (progress bars)

## Code Structure

All stage scripts follow identical structure for maintainability:

1. **Configuration** - Paths, parameters, thresholds
2. **Setup** - Directory creation, logging
3. **Functions** - `compute_rms()`, `process_audio_file()`
4. **Main Processing** - Metadata loading, file processing
5. **Summary** - Detailed statistics report

## Citation

If you use this pipeline or dataset, please cite the original sources:

```bibtex
@inproceedings{birdvox2018,
  title={BirdVox-DCASE-20k: Chirp detection in the presence of environmental sounds},
  author={Lostanlen, Vincent and Salamon, Justin and Cartwright, Mark and McFee, Brian and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
  booktitle={DCASE Workshop},
  year={2018}
}

@dataset{freefield2018,
  title={Freefield1010 Bird Detection Dataset},
  author={Stowell, Dan and Wood, Michael and Pamuła, Hanna and Stylianou, Yannis and Glotin, Hervé},
  year={2018}
}

@dataset{warblr2018,
  title={Warblrb10k Bird Detection Dataset},
  author={Stowell, Dan and Wood, Michael and Pamuła, Hanna and Stylianou, Yannis and Glotin, Hervé},
  year={2018}
}

@dataset{esc50,
  title={ESC-50: Dataset for Environmental Sound Classification},
  author={Piczak, Karol J},
  year={2015}
}

@article{bandara2023forest,
  title={Forest sound classification dataset: Fsc22},
  author={Bandara, Meelan and Jayasundara, Roshinie and Ariyarathne, Isuru and Meedeniya, Dulani and Perera, Charith},
  journal={Sensors},
  volume={23},
  number={4},
  pages={2032},
  year={2023},
  publisher={MDPI}
}

@article{fredianelli2025environmental,
  title={Environmental Noise Dataset for Sound Event Classification and Detection},
  author={Fredianelli, Luca and Artuso, Francesco and Pompei, Geremia and Licitra, Gaetano and Iannace, Gino and Akbaba, Andac},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1712},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

  GNU nano 7.2                                                                            17033970.bib                                                                                      
@misc{fredianelli_2025_17033970,
  author       = {Fredianelli, Luca and
                  Artuso, Francesco and
                  Pompei, Geremia and
                  Licitra, Gaetano and
                  Iannace, Gino and
                  Akbaba, Andaç},
  title        = {DataSEC - Dataset for Sound Event Classification
                   of environmental noise
                  },
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17033970},
  url          = {https://doi.org/10.5281/zenodo.17033970},
}
```

## License

This pipeline code is released under MIT License. Please respect the licenses of the original datasets.

## Related Work

This negative label curation is part of the **MyBAD** (My Bird Audio Dataset) project for tropical bird detection. See the main repository for positive label curation and model training.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note:** Dataset paths in scripts point to `/Volumes/Evo/datasets/`. Update these paths to match your local dataset locations before running.
