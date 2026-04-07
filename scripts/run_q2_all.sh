#!/bin/bash
# Run all Q2 experiments (Adversarial Attacks on CIFAR-10)
# Usage: ./scripts/run_q2_all.sh [--data_root ./data] [--use_wandb] [--dry_run]

set -e

# Ensure GPU-1 only (inside container, device=1 is mapped to CUDA:0)
export CUDA_VISIBLE_DEVICES=0

DATA_ROOT="./data"
USE_WANDB=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

run_cmd() {
    echo "Running: $*"
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

echo "============================================"
echo "Q2: Adversarial Attacks on CIFAR-10"
echo "============================================"

# Step 1: Train clean ResNet18 (target >=72% accuracy)
echo ""
echo "[Step 1/4] Training clean ResNet18..."
run_cmd python -m src.q2_attacks.train_clean_resnet18 \
    --data_root "$DATA_ROOT" \
    --epochs 30 \
    --batch_size 256 \
    --num_workers 2 \
    --output_dir results/q2/clean_resnet18 \
    --weights_path weights/q2/resnet18_clean_best.pt \
    $USE_WANDB

# Step 2: FGSM comparison (scratch vs ART)
echo ""
echo "[Step 2/4] Running FGSM attack comparison..."
run_cmd python -m src.q2_attacks.fgsm_compare \
    --data_root "$DATA_ROOT" \
    --weights_path weights/q2/resnet18_clean_best.pt \
    --eps 0.03 \
    --batch_size 512 \
    --num_workers 2 \
    --output_dir results/q2/fgsm_compare \
    $USE_WANDB

# Step 3: Train PGD adversarial detector
echo ""
echo "[Step 3/4] Training PGD adversarial detector..."
run_cmd python -m src.q2_detection.train_detector \
    --data_root "$DATA_ROOT" \
    --source_weights weights/q2/resnet18_clean_best.pt \
    --attack pgd \
    --batch_size 256 \
    --num_workers 2 \
    --epochs 15 \
    --output_dir results/q2/detectors \
    --weights_dir weights/q2/detectors \
    $USE_WANDB

# Step 4: Train BIM adversarial detector
echo ""
echo "[Step 4/4] Training BIM adversarial detector..."
run_cmd python -m src.q2_detection.train_detector \
    --data_root "$DATA_ROOT" \
    --source_weights weights/q2/resnet18_clean_best.pt \
    --attack bim \
    --batch_size 256 \
    --num_workers 2 \
    --epochs 15 \
    --output_dir results/q2/detectors \
    --weights_dir weights/q2/detectors \
    $USE_WANDB

# Step 5: Log all attack samples to WandB (if enabled)
if [ -n "$USE_WANDB" ]; then
    echo ""
    echo "[Step 5/5] Logging attack samples to WandB..."
    run_cmd python -m src.q2_attacks.log_all_attack_samples_wandb \
        --data_root "$DATA_ROOT" \
        --weights_path weights/q2/resnet18_clean_best.pt
fi

echo ""
echo "============================================"
echo "Q2 Complete! Results saved to results/q2/"
echo "============================================"
