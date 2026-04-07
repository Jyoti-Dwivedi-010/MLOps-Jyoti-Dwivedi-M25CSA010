import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.common import (
    accuracy_from_logits,
    cleanup_gpu,
    create_grad_scaler,
    ensure_dir,
    get_amp_context,
    get_device,
    maybe_init_wandb,
    register_cleanup_on_exit,
    save_json,
    set_seed,
)
from src.q2_attacks.data import get_cifar10_dataloaders
from src.q2_attacks.models import resnet18_cifar10



def run_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with get_amp_context(scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, y) * bs

    return total_loss / total, total_acc / total



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/q2/clean_resnet18")
    parser.add_argument("--weights_path", type=str, default="weights/q2/resnet18_clean_best.pt")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="assignment5-q2")
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(str(Path(args.weights_path).parent))

    device = get_device()
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision training
    scaler = create_grad_scaler(args.use_amp)

    run = maybe_init_wandb(args, args.wandb_project, "q2-clean-resnet18")

    rows = []
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True, scaler=scaler)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False, scaler=None)
        scheduler.step()

        row = {
            "epoch": epoch,
            "training_loss": tr_loss,
            "validation_loss": va_loss,
            "training_accuracy": tr_acc,
            "validation_accuracy": va_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        rows.append(row)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_acc={va_acc:.4f}")

        if run is not None:
            import wandb

            wandb.log(row)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), args.weights_path)

    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    te_loss, te_acc = run_epoch(model, test_loader, optimizer, criterion, device, train=False, scaler=None)

    table = pd.DataFrame(rows)
    table.to_csv(Path(args.output_dir) / "train_val_table.csv", index=False)
    payload = {
        "best_validation_accuracy": best_val,
        "test_accuracy_clean": te_acc,
        "test_loss_clean": te_loss,
        "best_weights": args.weights_path,
    }
    save_json(str(Path(args.output_dir) / "summary.json"), payload)

    print(f"\nTest accuracy: {te_acc:.4f} (target: >=72%)")
    if te_acc >= 0.72:
        print("✓ Target accuracy achieved!")
    else:
        print("✗ Target accuracy not achieved. Consider more epochs or tuning.")

    if run is not None:
        import wandb

        wandb.log({"test_accuracy_clean": te_acc, "test_loss_clean": te_loss})
        run.finish()

    # Final cleanup
    cleanup_gpu()
    print("Training complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
