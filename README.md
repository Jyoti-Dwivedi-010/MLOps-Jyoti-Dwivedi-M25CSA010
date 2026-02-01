# Lab 2: CIFAR-10 CNN Classification Worksheet
**Author:** Jyoti Dwivedi (M25CSA010)

## 1. Project Overview
Design and analysis of a custom 3-layer CNN on the CIFAR-10 dataset, focusing on computational efficiency and robust generalization using WandB tracking.

## 2. Model Architecture & Complexity
- **Architecture:** 3 Conv layers (32, 64, 128) + 3 MaxPool + 2 FC layers with Dropout (0.25).
- **FLOPs:** 10.85 Million (Calculated via thop).
- **Parameters:** 0.62 Million.

## 3. Findings & Performance
- **Final Test Accuracy:** 75.57%
- **Validation Accuracy:** 88.67%
- **Training Accuracy:** 84.29%
- **Observation:** Unlike standard overfitted models where validation accuracy plateaus low, this model maintains high validation performance (88%+), confirming that the 20-degree rotation augmentation effectively regularized the network.
- **Visualization:** Gradient flow and weight updates remained stable across all 30 epochs.

## 4. Links
- **WandB Link:** https://wandb.ai/dwivedijyoti620-prom-iit-rajasthan/cifar10-lab2/runs/6g0tde6x
- **GitHub Link:** https://github.com/Jyoti-Dwivedi-010/MLOps-Jyoti-Dwivedi-M25CSA010/tree/Jyoti-Dwivedi_M25CSA010_Lab2_worksheet
