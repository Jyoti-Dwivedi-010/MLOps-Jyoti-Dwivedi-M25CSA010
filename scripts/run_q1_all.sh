#!/bin/bash
# Run all Q1 experiments (ViT-S + LoRA on CIFAR-100)
# Usage: ./scripts/run_q1_all.sh [--data_root ./data] [--use_wandb] [--dry_run]

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
echo "Q1: ViT-S + LoRA on CIFAR-100"
echo "============================================"

# Step 1: Baseline (head-only fine-tuning, no LoRA)
echo ""
echo "[Step 1/3] Training baseline (head-only, no LoRA)..."
run_cmd python -m src.q1_vit_lora.train_baseline \
    --data_root "$DATA_ROOT" \
    --epochs 10 \
    --batch_size 128 \
    --num_workers 2 \
    --output_dir results/q1/baseline \
    $USE_WANDB

# Step 2: LoRA grid experiments (rank={2,4,8}, alpha={2,4,8})
echo ""
echo "[Step 2/3] Running LoRA grid experiments..."
run_cmd python -m src.q1_vit_lora.run_grid \
    --data_root "$DATA_ROOT" \
    --epochs 10 \
    --batch_size 128 \
    --num_workers 2 \
    --output_root results/q1/lora_grid \
    --weights_root weights/q1/grid \
    $USE_WANDB

# Step 3: Optuna hyperparameter search
echo ""
echo "[Step 3/3] Running Optuna search for best LoRA hyperparameters..."
run_cmd python -m src.q1_vit_lora.optuna_lora_search \
    --data_root "$DATA_ROOT" \
    --batch_size 128 \
    --num_workers 2 \
    --n_trials 12 \
    --search_epochs 3 \
    --output_dir results/q1/optuna

echo ""
echo "============================================"
echo "Q1 Complete! Results saved to results/q1/"
echo "============================================"
