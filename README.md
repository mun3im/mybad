# MyBAD: Malaysian Bird Activity Detection Lightweight Edge AI Models **DRAFT**

Welcome to the official companion repository for the MyBAD (Malaysian Bird Activity Detection) dataset! 

This project demonstrates how to train, benchmark, and deploy lightweight neural networks for real-time bird activity detection on Edge AI hardware.

MyBAD contains 50,000 three-second audio clips (25k positive / 25k negative) from Malaysia provided as 16kHz 3-second clips.

The goal of this project is to make real-time, low-power bird monitoring practical and accessible — whether you’re targeting microcontrollers, single-board computers, or embedded AI accelerators.

### 🚀 Lightweight BAD Models

Implementations of tiny and efficient neural networks optimized for MyBAD, including:

- TODO
#- **MobileNetV3-Small** Original MCU target
#- **MyBAD-baseline** 99.99% same as TinyChirp CNNMel
#- **MyBAD-separable** Depthwise separable
#- **MyBAD-batchnorm** Baseline + batch normalization
#- **MyBAD-separablebatchnorm** Depthwise separable + batch normalization
#- **MyBAD-optimized** Large, optimized for accuracy


All models are designed to run smoothly at low latency and minimal power, even on MCUs.

## 🎧 Dataset Overview


Positive samples come from Malaysian bird recordings on Xeno-canto.
Negative samples are drawn from BirdCLEF BAD sources (BirdVox, Freefield1010, Warblr), with extra non-bird environmental sounds from ESC50, FSC22 and DataSEC.


## 🧪 Training & Experiments

This repository includes:

###TODO
- Training pipelines for all model families
- Baseline accuracy benchmarks for each spectrogram size
- Ablation experiments (model width, mel bins, filter sizes, quantization)
- Visualization tools for confusion matrices and ROC curves

You can reproduce the full BAD benchmark suite with a single command.

## ⚡ Edge AI Deployment

We provide export scripts for:

- TensorFlow Lite (int8 / float32)
- TFLM (TensorFlow Lite for Microcontrollers)

And yes — there’s a folder of example firmware templates for STM32 devices.


## 📦 Folder Structure

```
/models          # Implementations of TinyCNN, DS-CNN, MobileNetV3-Tiny, etc.
/data            # Loaders for the MyBAD mel-spectrogram .npy files
/train           # Training and validation scripts
/export          # TFLite, ONNX, Edge TPU export utilities
/experiments     # Ablation studies & benchmark configs
/deployment      # Example firmware for MCU inference
```

## 🔍 Why This Matters

Tropical regions are critically underrepresented in global bioacoustics research.
With Edge AI bird activity detection, we can build:

- a balanced detection dataset,
- microcontroller-friendly spectrograms, and
- lightweight deployable models

makes MyBAD a powerful platform for building scalable, real-time biodiversity sensors, even in remote forest environments.

## 📜 Citation

If you use this repository or the MyBAD dataset in your research, please cite the Zenodo dataset: 

M. M. A. Zabidi, “MyBAD: Malaysian Bird Activity Detection Dataset”. Zenodo, Dec. 02, 2025. DOI https://doi.org/10.5281/zenodo.17791820

