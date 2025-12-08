# MyBAD: Malaysian Bird Activity Detection — Lightweight Edge AI Models

Welcome to the official companion repository for the MyBAD (Malaysian Bird Activity Detection) dataset! 🎉
This repo shows how to train, evaluate, and deploy ultra-lightweight neural networks for bird activity detection on edge devices, using the 56,000 curated 3-second audio clips from Malaysia and Singapore.

The goal of this project is to make real-time, low-power bird monitoring practical and accessible — whether you’re targeting microcontrollers, single-board computers, or embedded AI accelerators.

##🌿 What This Repository Provides
###🚀 Lightweight BAD Models

Implementations of tiny and efficient neural networks optimized for MyBAD, including:

- **TinyCNN-BAD** (depthwise separable 1D/2D variants)
- **MobileNetV3-Tiny** (compressed from Small)
- **DS-CNN** (Google Edge TPU style)
- **TC-ResNet8 / TC-ResNet14** (Temporally Convolutional CNNs)
- **Micro-Attention BADNet** (≤150k parameters)


All models are designed to run smoothly at low latency and minimal power, even on MCUs.

🎧 Dataset Integration

The codebase loads the .npy mel-spectrograms (80×184, 64×184, 48×184, 32×184, 16×184) directly, allowing fast experimentation across different input sizes.
Just download the dataset from Zenodo, set the path in config.json, and you're good to go.

🧪 Training & Experiments

This repository includes:

Training pipelines for all model families

Baseline accuracy benchmarks for each spectrogram size

Ablation experiments (model width, mel bins, filter sizes, quantization)

Visualization tools for confusion matrices and ROC curves

You can reproduce the full BAD benchmark suite with a single command.

⚡ Edge AI Deployment

We provide export scripts for:

TensorFlow Lite (int8 / float16)

TFLM (TensorFlow Lite for Microcontrollers)

ONNX for embedded Linux devices

Edge TPU-compatible models

And yes — there’s a folder of example firmware templates for ESP32-S3 and STM32 devices.
