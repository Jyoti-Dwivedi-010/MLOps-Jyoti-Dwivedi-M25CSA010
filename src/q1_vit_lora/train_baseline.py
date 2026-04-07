import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.common import (
    accuracy_from_logits,
    cleanup_gpu,
    count_trainable_params,
    count_total_params,
    create_grad_scaler,
    dump_config,
    ensure_dir,
    get_amp_context,
    get_device,
    maybe_init_wandb,
    register_cleanup_on_exit,
    save_json,
    set_seed,
)
from src.q1_vit_lora.data import get_cifar100_dataloaders
from src.q1_vit_lora.modeling import create_vit_small_for_cifar100, freeze_all_but_head



def run_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for images, labels in tqdm(loader, leave=False):
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/q1/baseline")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="assignment5-q1")
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    dump_config(str(Path(args.output_dir) / "config.json"), args)

    device = get_device()
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_cifar100_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_vit_small_for_cifar100().to(device)
    freeze_all_but_head(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = create_grad_scaler(args.use_amp)

    run = maybe_init_wandb(args, args.wandb_project, "q1-baseline-head-only")

    rows = []
    best_val = -1.0
    best_path = Path(args.output_dir) / "best_baseline_head_only.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True, scaler=scaler)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False, scaler=None)

        row = {
            "epoch": epoch,
            "training_loss": tr_loss,
            "validation_loss": va_loss,
            "training_accuracy": tr_acc,
            "validation_accuracy": va_acc,
        }
        rows.append(row)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_acc={va_acc:.4f}")

        if run is not None:
            import wandb

            wandb.log(row)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)

    table = pd.DataFrame(rows)
    table.to_csv(Path(args.output_dir) / "train_val_table.csv", index=False)

    params_payload = {
        "trainable_parameters": count_trainable_params(model),
        "total_parameters": count_total_params(model),
        "best_validation_accuracy": best_val,
        "best_weights": str(best_path),
    }
    save_json(str(Path(args.output_dir) / "summary.json"), params_payload)

    if run is not None:
        run.finish()

    # Final cleanup
    cleanup_gpu()
    print("Training complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
