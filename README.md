# MyBAD: Malaysian Bird Activity Detection Lightweight Edge AI Models 

Welcome to the official companion repository for the MyBAD (Malaysian Bird Activity Detection) dataset! 

This project demonstrates how to train, benchmark, and deploy lightweight neural networks for real-time bird activity detection on Edge AI hardware.

MyBAD contains 50,000 three-second audio clips (25k positive / 25k negative) from Malaysia, processed into multiple mel-spectrogram resolutions and packaged for fast ML experimentation.

The goal of this project is to make real-time, low-power bird monitoring practical and accessible — whether you’re targeting microcontrollers, single-board computers, or embedded AI accelerators.

## 🌿 Features

- 🔊 Direct support for MyBAD .npy mel-spectrograms

- ⚡ Lightweight CNN architectures tailored for edge devices

- 🧪 Full training + evaluation pipelines

- 📉 Ablation studies for model size, mel resolution, and quantization

- 📦 Export tools for TFLite, TFLM, ONNX, and Edge TPU

- 🛠 Example firmware templates for MCUs (ESP32-S3, STM32)

### 🚀 Lightweight BAD Models

Implementations of tiny and efficient neural networks optimized for MyBAD, including:

- **TinyCNN-BAD** (depthwise separable 1D/2D variants)
- **MobileNetV3-Tiny** (compressed from Small)
- **DS-CNN** (Google Edge TPU style)
- **TC-ResNet8 / TC-ResNet14** (Temporally Convolutional CNNs)
- **Micro-Attention BADNet** (≤150k parameters)


All models are designed to run smoothly at low latency and minimal power, even on MCUs.

## 🎧 Dataset Overview

The dataset provides mel-spectrograms in five resolutions:


- 80 × 184

- 64 × 184

- 48 × 184

- 32 × 184

- 16 × 184

Each sample is stored as a NumPy .npy file, ready for direct loading into PyTorch or TensorFlow.

Positive samples come from Malaysian/Singaporean bird recordings on Xeno-canto, plus supplements from Macaulay Library for common species.
Negative samples are drawn from BirdCLEF BAD sources (BirdVox, Freefield1010, Warblr), with extras from Xeno-canto, ESC, and FSC.

Download the dataset here:

## 🧪 Training & Experiments

This repository includes:

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

/models          # Implementations of TinyCNN, DS-CNN, MobileNetV3-Tiny, etc.
/data            # Loaders for the MyBAD mel-spectrogram .npy files
/train           # Training and validation scripts
/export          # TFLite, ONNX, Edge TPU export utilities
/experiments     # Ablation studies & benchmark configs
/deployment      # Example firmware for MCU inference


## 🔍 Why This Matters

Tropical regions are critically underrepresented in global bioacoustics research.
With Edge AI bird activity detection, we can build:

- a balanced detection dataset,

- microcontroller-friendly spectrograms, and

- lightweight deployable models

makes MyBAD a powerful platform for building scalable, real-time biodiversity sensors, even in remote forest environments.

## 📜 Citation

If you use this repository or the MyBAD dataset in your research, please cite the Zenodo dataset:

