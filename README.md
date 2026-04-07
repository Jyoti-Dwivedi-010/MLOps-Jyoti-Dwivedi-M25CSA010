# Assignment-5 (Docker + Python + PyTorch)

This repository is structured to satisfy Assignment-5 requirements for:
- Q1: ViT-S on CIFAR-100, baseline head-only fine-tuning and LoRA (PEFT) experiments.
- Q2: Adversarial attacks and adversarial detection on CIFAR-10 using IBM ART.

## 1) Repository Structure

- src/q1_vit_lora: Q1 training, LoRA grid, Optuna search, HuggingFace upload helper.
- src/q2_attacks: Q2 clean classifier training, FGSM from scratch and ART comparison, WandB image logging.
- src/q2_detection: Q2 adversarial detector training for PGD and BIM.
- scripts: Bash helper scripts to run complete pipelines (Linux).
- results: CSV/JSON tables and generated plots.
- weights: trained model checkpoints.
- reports: final PDF report (naming example: B22CS043_Firstname_Surname_Ass5.pdf).

## 2) Branch and Submission Checklist

1. Create branch named: Assignment 5
2. Push:
   - requirements.txt
   - all .py files
   - report PDF
   - best model weights for Q1
   - all weights for Q2
3. Add in README and report:
   - WandB links
   - HuggingFace link
   - train-val tables and graphs for Q1 and Q2
   - Q2 qualitative original/adversarial image samples
4. Upload on Classroom:
   - GitHub branch link
   - WandB link
   - HuggingFace link
   - report PDF only

## 3) Environment Setup

### GPU Configuration

This project is configured to use **GPU-1 only** on a shared GPU server (48GB).
The Docker container and all scripts are set to use only GPU-1 via `CUDA_VISIBLE_DEVICES`.

### Docker (required by assignment)

**Prerequisite:** Ensure Docker daemon is running with GPU support (nvidia-docker).

1. Build container:
   ```bash
   docker build -t assignment5-mlops .
   ```

2. Run container (uses GPU-1 only with increased shared memory):
   ```bash
   docker run --gpus '"device=1"' --rm -it --shm-size=8g --ipc=host -v "$(pwd):/workspace" assignment5-mlops bash
   ```

**Bash helper script:**
```bash
chmod +x scripts/run_in_docker.sh
./scripts/run_in_docker.sh
```

### Local Setup (without Docker)

1. Create a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv .venv
   ```

2. Activate it:
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Set GPU (ensure GPU-1 is used):
   ```bash
   export CUDA_VISIBLE_DEVICES=1
   ```

## 4) Q1 Commands (ViT-S + LoRA)

**Run all Q1 experiments at once:**
```bash
chmod +x scripts/run_q1_all.sh
./scripts/run_q1_all.sh --data_root ./data --use_wandb
```

### 4.1 Baseline (without LoRA, classification head fine-tune)

```bash
python -m src.q1_vit_lora.train_baseline \
    --data_root ./data \
    --epochs 10 \
    --batch_size 128 \
    --output_dir results/q1/baseline \
    --use_wandb
```

Outputs:
- results/q1/baseline/train_val_table.csv
- results/q1/baseline/summary.json

### 4.2 LoRA Grid Experiments (rank in {2,4,8}, alpha in {2,4,8}, dropout=0.1)

```bash
python -m src.q1_vit_lora.run_grid \
    --data_root ./data \
    --epochs 10 \
    --batch_size 128 \
    --output_root results/q1/lora_grid \
    --weights_root weights/q1/grid \
    --use_wandb
```

Outputs per experiment:
- train_val_table.csv
- summary.json
- classwise_test_accuracy_hist.png
- lora_gradient_updates.png

Combined test table:
- results/q1/lora_grid/q1_test_results_table.csv

### 4.3 Optuna Search (LoRA hyperparameters only)

```bash
python -m src.q1_vit_lora.optuna_lora_search \
    --data_root ./data \
    --n_trials 12 \
    --search_epochs 3 \
    --output_dir results/q1/optuna
```

Output:
- results/q1/optuna/best_lora_optuna.json

### 4.4 Upload Best Q1 Model to HuggingFace

```bash
python -m src.q1_vit_lora.upload_best_to_hf \
    --repo_id <username/repo_name> \
    --token <hf_token> \
    --weights_path <best_weight_path> \
    --rank <r> \
    --alpha <a> \
    --dropout 0.1
```

### 4.5 [Optional] Partial Freeze Experiment

```bash
python -m src.q1_vit_lora.train_partial_freeze \
    --data_root ./data \
    --batch_size 128 \
    --num_workers 2 \
    --epochs 10 \
    --rank 4 \
    --alpha 4 \
    --dropout 0.1 \
    --trainable_blocks 2 \
    --output_dir results/q1/partial_freeze \
    --save_weights_dir weights/q1/partial_freeze \
    --use_wandb
```

Output:
- results/q1/partial_freeze/train_val_table.csv
- results/q1/partial_freeze/summary.json
- results/q1/partial_freeze/classwise_test_accuracy_hist.png
- results/q1/partial_freeze/lora_gradient_updates.png

## 5) Q2 Commands (Adversarial Attacks + Detection)

**Run all Q2 experiments at once:**
```bash
chmod +x scripts/run_q2_all.sh
./scripts/run_q2_all.sh --data_root ./data --use_wandb
```

### 5.1 Train clean ResNet18 (target >=72% test accuracy)

```bash
python -m src.q2_attacks.train_clean_resnet18 \
    --data_root ./data \
    --epochs 30 \
    --batch_size 256 \
    --output_dir results/q2/clean_resnet18 \
    --weights_path weights/q2/resnet18_clean_best.pt \
    --use_wandb
```

Output:
- results/q2/clean_resnet18/train_val_table.csv
- results/q2/clean_resnet18/summary.json

### 5.2 FGSM comparison (Scratch vs ART)

```bash
python -m src.q2_attacks.fgsm_compare \
    --data_root ./data \
    --weights_path weights/q2/resnet18_clean_best.pt \
    --eps 0.03 \
    --batch_size 512 \
    --output_dir results/q2/fgsm_compare \
    --use_wandb
```

Outputs:
- results/q2/fgsm_compare/accuracy_comparison.csv
- results/q2/fgsm_compare/summary.json
- results/q2/fgsm_compare/fgsm_clean_vs_scratch_vs_art.png

### 5.3 Adversarial detector (ResNet34) for PGD and BIM

```bash
# PGD detector
python -m src.q2_detection.train_detector \
    --data_root ./data \
    --source_weights weights/q2/resnet18_clean_best.pt \
    --attack pgd \
    --batch_size 256 \
    --output_dir results/q2/detectors \
    --weights_dir weights/q2/detectors \
    --use_wandb

# BIM detector
python -m src.q2_detection.train_detector \
    --data_root ./data \
    --source_weights weights/q2/resnet18_clean_best.pt \
    --attack bim \
    --batch_size 256 \
    --output_dir results/q2/detectors \
    --weights_dir weights/q2/detectors \
    --use_wandb
```

Outputs:
- results/q2/detectors/pgd/train_val_table.csv
- results/q2/detectors/pgd/summary.json
- results/q2/detectors/bim/train_val_table.csv
- results/q2/detectors/bim/summary.json

### 5.4 Log 10 sample pairs for FGSM(scratch), FGSM(ART), PGD, BIM on WandB

```bash
python -m src.q2_attacks.log_all_attack_samples_wandb \
    --data_root ./data \
    --weights_path weights/q2/resnet18_clean_best.pt \
    --wandb_project assignment5-q2
```

## 6) One-Command Pipeline Helpers

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Q1 full (inside Docker container or with CUDA_VISIBLE_DEVICES=1):
./scripts/run_q1_all.sh --data_root ./data --use_wandb

# Q2 full:
./scripts/run_q2_all.sh --data_root ./data --use_wandb

# Dry run (print commands only):
./scripts/run_q1_all.sh --data_root ./data --use_wandb --dry_run
./scripts/run_q2_all.sh --data_root ./data --use_wandb --dry_run

# GPU memory cleanup:
./scripts/cleanup_gpu.sh
```

Note:
- If you omit --use_wandb, scripts still run and save local results.
- The Q2 sample logging script is only triggered when --use_wandb is provided.

## 7) GPU Memory Management

All scripts automatically clean up GPU memory on exit. For manual cleanup:
```bash
./scripts/cleanup_gpu.sh
```

Or in Python:
```python
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
```

## 8) Q1 Results

### Q1 Test Results Table

| LoRA Layers | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|-------------|------|-------|---------|---------------|------------------|
| without     | -    | -     | -       | 81.16%        | 38,500           |
| with        | 2    | 2     | 0.1     | 88.97%        | 113,864          |
| with        | 2    | 4     | 0.1     | 89.20%        | 113,864          |
| with        | 2    | 8     | 0.1     | 89.27%        | 113,864          |
| with        | 4    | 2     | 0.1     | 88.99%        | 150,728          |
| with        | 4    | 4     | 0.1     | 89.06%        | 150,728          |
| with        | 4    | 8     | 0.1     | 89.26%        | 150,728          |
| with        | 8    | 2     | 0.1     | 89.11%        | 224,456          |
| with        | 8    | 4     | 0.1     | 89.27%        | 224,456          |
| **with**    | **8**| **8** | **0.1** | **89.33%**    | **224,456**      |

**Best Model**: Rank=8, Alpha=8, Dropout=0.1 → **89.33% Test Accuracy**

### Optuna Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rank      | 4     |
| Alpha     | 4     |
| Learning Rate | 0.00048 |
| Validation Accuracy | 89.46% |

### [Optional] Partial Freeze Experiment

Configuration: Keep last 2 transformer blocks trainable (no LoRA), apply LoRA to frozen blocks (0-9).

| Experiment | Trainable Blocks | LoRA Blocks | Rank | Alpha | Test Accuracy | Trainable Params |
|------------|------------------|-------------|------|-------|---------------|------------------|
| Full Freeze + LoRA | 0 | 12 | 4 | 4 | 89.06% | 150,728 |
| Partial Freeze + LoRA | 2 | 10 | 4 | 4 | 88.40% | 3,648,868 |

**Analysis**: The partial freeze approach (keeping last 2 blocks trainable) resulted in slightly lower accuracy (88.40% vs 89.06%) despite having ~24x more trainable parameters. This suggests that LoRA on frozen layers is more parameter-efficient and the pre-trained representations in the last layers are already well-suited for the downstream task. The full-freeze LoRA approach maintains the generalization of pre-trained weights while efficiently adapting to CIFAR-100.

### Q1 Generated Files

- `results/q1/lora_grid/q1_test_results_table.csv` - Combined test results
- `results/q1/lora_grid/r*_a*/train_val_table.csv` - Per-experiment training logs
- `results/q1/lora_grid/r*_a*/classwise_test_accuracy_hist.png` - Class-wise accuracy histograms
- `results/q1/lora_grid/r*_a*/lora_gradient_updates.png` - Gradient update graphs
- `results/q1/optuna/best_lora_optuna.json` - Optuna search results
- `results/q1/partial_freeze/summary.json` - Partial freeze experiment results

## Q2 Results

### Q2(i) FGSM Attack Comparison

| Metric | Clean | FGSM Scratch | FGSM ART |
|--------|-------|--------------|----------|
| Accuracy | 83.26% | 9.59% | 19.06% |
| Drop from Clean | - | 73.67% | 64.20% |

**Analysis**: Both FGSM implementations successfully fool the model, with scratch implementation being slightly more effective (lower accuracy = stronger attack). The ART implementation provides a standardized baseline for comparison.

### Q2(ii) Adversarial Detection Results

| Attack Type | Test Detection Accuracy | Target | Status |
|-------------|------------------------|--------|--------|
| PGD | 78.71% | ≥70% | ✅ |
| BIM | 91.48% | ≥70% | ✅ |

**Analysis**: The BIM detector achieved higher accuracy (91.48%) compared to PGD (78.71%). This could be because BIM generates more distinguishable perturbations that are easier to detect.

### Q2 Generated Files

- `results/q2/clean_resnet18/train_val_table.csv` - Clean model training logs
- `results/q2/clean_resnet18/summary.json` - Clean model summary
- `results/q2/fgsm_compare/accuracy_comparison.csv` - FGSM comparison results
- `results/q2/fgsm_compare/fgsm_clean_vs_scratch_vs_art.png` - Visual comparison
- `results/q2/detectors/pgd/train_val_table.csv` - PGD detector training
- `results/q2/detectors/bim/train_val_table.csv` - BIM detector training
- Attack samples logged to WandB (10 samples each for FGSM, PGD, BIM)

## 9) WandB and HuggingFace Links

- **WandB Q1**: https://wandb.ai/m25csa010-iit-jodhpur/assignment5-q1
- **WandB Q2**: https://wandb.ai/m25csa010-iit-jodhpur/assignment5-q2
- **HuggingFace Q1 best model**: https://huggingface.co/JD16112001/vit-lora-cifar100-best

## 10) GitHub Page Update

Add this assignment repository/branch and final report links to your GitHub profile page or assignment index.
