# ML Text Classification Pipeline (Docker + Hugging Face)

End-to-end transformer fine-tuning pipeline with reproducible Docker deployment and Hugging Face model hosting.

---

## Overview

This project implements:

- Notebook → production Python scripts
- Transformer fine-tuning using Hugging Face `Trainer`
- Evaluation with Accuracy and Weighted F1
- Separate Docker images for training and inference
- Model hosting on Hugging Face Hub
- Reproducible evaluation from remote model

---

## Model

Base model: `prajjwal1/bert-tiny`

- 2 Transformer layers
- Hidden size: 128
- 8-class single-label classification

Classes:
- romance
- history_biography
- comics_graphic
- poetry
- fantasy_paranormal
- mystery_thriller_crime
- children
- young_adult


Note: Sequence length reduced to 128 due to CPU constraints.

---

## Results

| Metric | Before Training | After Training |
|--------|-----------------|----------------|
| Eval Loss | 3.0326 | 2.0885 |
| Accuracy | 7.44% | 10.25% |
| Weighted F1 | 0.0695 | 0.1276 |

---

## Project Structure
├── data.py
├── train.py
├── evaluate.py
├── requirements.txt
├── Dockerfile.train
├── Dockerfile.eval
└── README.md

---

## Docker Usage

### Build Training Image

- docker build -t assignment3-train -f Dockerfile.train .
- docker run assignment3-train python /src/train.py

### Build Evaluation Image

- docker build -t assignment3-eval -f Dockerfile.eval .
- docker run -e HF_TOKEN=**** assignment3-eval python /src/evaluate.py
---
### Dependencies

torch==2.1.2
transformers==4.36.2
datasets==2.16.1
scikit-learn==1.3.2
accelerate==0.25.0

### Hugging Face Model

- https://huggingface.co/JD16112001/model_assignment3
---
### Key Takeaways

Reproducible ML pipeline using Docker

Clean separation of training and inference environments

Remote model deployment via Hugging Face Hub

CPU-constrained fine-tuning workflow
