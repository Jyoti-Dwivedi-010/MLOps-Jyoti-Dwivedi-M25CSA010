import argparse
from pathlib import Path

import optuna
import torch
from torch import nn

from src.common import (
    accuracy_from_logits,
    cleanup_gpu,
    create_grad_scaler,
    ensure_dir,
    get_amp_context,
    get_device,
    register_cleanup_on_exit,
    save_json,
    set_seed,
)
from src.q1_vit_lora.data import get_cifar100_dataloaders
from src.q1_vit_lora.modeling import apply_lora_to_vit, create_vit_small_for_cifar100



def run_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with get_amp_context(scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, labels) * bs

    return total_loss / total, total_acc / total



def objective(trial, args, train_loader, val_loader, device):
    rank = trial.suggest_categorical("rank", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)

    model = create_vit_small_for_cifar100()
    for p in model.parameters():
        p.requires_grad = False
    model = apply_lora_to_vit(model, rank, alpha, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = create_grad_scaler(True)

    best_val = 0.0
    for epoch in range(args.search_epochs):
        run_epoch(model, train_loader, optimizer, criterion, device, train=True, scaler=scaler)
        _, va_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False, scaler=None)
        best_val = max(best_val, va_acc)
        
        # Report intermediate values for pruning
        trial.report(va_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Cleanup after each trial
    del model, optimizer, scaler
    cleanup_gpu()
    
    return best_val



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--search_epochs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/q1/optuna")
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = get_device()
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_cifar100_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(t, args, train_loader, val_loader, device),
        n_trials=args.n_trials,
    )

    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    save_json(str(Path(args.output_dir) / "best_lora_optuna.json"), payload)
    
    print(f"\nOptuna search complete!")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Final cleanup
    cleanup_gpu()
    print("GPU memory cleaned up.")



if __name__ == "__main__":
    main()
