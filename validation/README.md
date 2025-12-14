# MyBAD Dataset Validation

Baseline performance evaluation of the MyBAD (Malaysian Bird Activity Detection) dataset using state-of-the-art deep learning architectures.

## Overview

This repository contains validation experiments demonstrating the quality and usability of the MyBAD dataset for bird activity detection tasks. We evaluate four CNN architectures to establish baseline performance metrics and validate dataset characteristics.

## Dataset

**MyBAD** is a balanced binary classification dataset for Malaysian bird activity detection:

- **50,000 audio clips** (3 seconds each, 16kHz)
  - 25,000 positive samples (Malaysian bird vocalizations)
  - 25,000 negative samples (background/environmental noise)
- **High-resolution spectrograms**: 224×224 mel-spectrograms
- **80/10/10 split**: Train (40K) / Validation (5K) / Test (5K)

## Validation Results

All models achieved exceptional performance, demonstrating high dataset quality:

| Model | Accuracy | AUC | F1 Score | Training Time | Parameters |
|-------|----------|-----|----------|---------------|------------|
| **MobileNetV3-Small** | **99.84%** | 99.96% | 99.84% | 1h 12m | 1.1M |
| **ResNet50** | 99.64% | 99.94% | 99.64% | 4h 28m | 24.2M |
| **VGG16** | 99.54% | **99.99%** | 99.54% | 5h 38m | 14.9M |
| **EfficientNetB0** | 98.72% | 99.79% | 98.71% | 3h 22m | 4.4M |

### Key Performance Indicators

- **Average Accuracy**: 99.44% across all architectures
- **Best Overall**: MobileNetV3-Small (99.84% accuracy, fastest training)
- **Consistent Performance**: Only 1.12% variance between models
- **Balanced Metrics**: High precision and recall across all models

### Detailed Metrics (MobileNetV3-Small)

```
Accuracy:  99.84%
AUC:       99.96%
Precision: 99.72%
Recall:    99.96%
F1 Score:  99.84%
```

Perfect for deployment scenarios requiring:
- ✅ Fast inference (lightweight model)
- ✅ High accuracy (>99%)
- ✅ Low false negatives (99.96% recall)

## Architecture Details

### Models Evaluated

1. **MobileNetV3-Small** - Efficient mobile-optimized CNN
2. **ResNet50** - Deep residual network with skip connections
3. **VGG16** - Classic deep CNN architecture
4. **EfficientNetB0** - Compound-scaled efficient network

### Training Configuration

- **Framework**: TensorFlow 2.15
- **Hardware**: CPU-only (accessible to most researchers)
- **Optimizer**: Adam with Cosine Decay learning rate
- **Regularization**: L2 (1e-4) + Dropout (0.5)
- **Early Stopping**: Patience of 15 epochs
- **Batch Size**: 32

### Preprocessing

Audio clips are converted to high-resolution mel-spectrograms:
- **n_mels**: 224 frequency bins
- **n_hops**: 224 time steps
- **n_fft**: 512
- **Input shape**: 224×224×1 (grayscale)
- **Preprocessing**: Log-mel transformation with per-sample normalization

## Key Findings

### Dataset Quality

✅ **Exceptional Performance** - All models >98.7% accuracy indicates:
- High-quality, consistent annotations
- Clear acoustic signatures for bird vocalizations
- Minimal label noise

✅ **Architectural Consistency** - Low variance across different architectures (1.12%) demonstrates:
- Dataset quality independent of model choice
- Robust discriminative features
- Well-balanced class distribution

✅ **Computational Efficiency** - Strong results on CPU-only training proves:
- Accessibility for researchers with limited resources
- Fast iteration for model development
- Practical for real-world deployment

### Performance Analysis

**Why MobileNetV3-Small Excels:**
- Achieves highest accuracy with smallest model size
- 6× faster training than VGG16
- 20× fewer parameters than ResNet50
- Ideal for edge deployment scenarios

**Trade-offs:**
- Larger models (ResNet50, VGG16) offer minimal accuracy gains
- Training time increases significantly with model size
- MobileNetV3 provides best accuracy/efficiency balance

## Repository Structure

```
validation/
├── train_mybad_classic.py       # Main training script
├── utils.py                      # Plotting utilities
├── deep_analysis.py              # Statistical analysis
├── results_*/                    # Experiment results
│   ├── results.txt              # Performance metrics
│   ├── stats.txt                # Dataset statistics
│   └── *.png                    # Visualization plots
├── VALIDATION_EXPERIMENT.md      # Detailed methodology
└── README.md                     # This file
```

## Quick Start

### Requirements

```bash
# Create environment
conda create -n mybad python=3.10
conda activate mybad

# Install dependencies
pip install tensorflow==2.15
pip install librosa numpy pandas scikit-learn matplotlib seaborn tqdm
```

### Run Validation

```bash
# Single model training
python train_mybad_classic.py \
    --model mobilenetv3s \
    --dataset_dir /path/to/mybad

# Available models: mobilenetv3s, resnet50, vgg16, efficientnetb0
```

### Analyze Results

```bash
# Generate comprehensive analysis
python deep_analysis.py

# Outputs:
# - analysis_plots/          # Comparison visualizations
# - summary_statistics.csv   # Aggregated metrics
# - analysis_table.tex       # LaTeX table for papers
```

## Citation

If you use the MyBAD dataset or these validation results, please cite:

```bibtex
@dataset{zabidi_2025_17913039,
  author       = {Zabidi, Muhammad Mun'im A.},
  title        = {MyBAD: Malaysian Bird Activity Detection Dataset},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17913039},
  url          = {https://doi.org/10.5281/zenodo.17913039}
}
```

## Applications

The MyBAD dataset is suitable for:

- **Biodiversity Monitoring** - Automated bird species presence detection
- **Acoustic Scene Classification** - Natural soundscape analysis
- **Transfer Learning** - Pre-training for other bioacoustic tasks
- **Model Benchmarking** - Standardized evaluation of audio classifiers
- **Edge AI Development** - Testing lightweight models for field deployment

## Future Work

Potential extensions of this validation study:

- Multi-class classification (species-level identification)
- Cross-dataset generalization testing
- Real-time inference optimization
- Few-shot learning evaluation
- Domain adaptation from other bird datasets

## Acknowledgments

This validation study was conducted using TensorFlow 2.15 on CPU resources. Spectrogram preprocessing utilizes the librosa library for audio analysis.

## License

Dataset and code license information available in the main MyBAD repository.

---

**Questions or Issues?** Please open an issue in the main MyBAD repository or contact the dataset author.
