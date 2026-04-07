import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.common import (
    accuracy_from_logits,
    cleanup_gpu,
    compute_per_class_accuracy,
    count_trainable_params,
    count_total_params,
    create_grad_scaler,
    dump_config,
    ensure_dir,
    get_amp_context,
    get_device,
    grad_norm,
    maybe_init_wandb,
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
    grad_values = []

    lora_params = [
        p
        for n, p in model.named_parameters()
        if ("lora_" in n or "lora" in n.lower()) and p.requires_grad
    ]

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
            grad_values.append(grad_norm(lora_params))

        bs = images.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, labels) * bs

    grad_mean = float(sum(grad_values) / max(len(grad_values), 1))
    return total_loss / total, total_acc / total, grad_mean



def eval_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)



def plot_per_class_histogram(per_class_acc, out_path: str):
    plt.figure(figsize=(14, 5))
    plt.bar(list(range(len(per_class_acc))), per_class_acc)
    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.title("Class-wise Test Accuracy Histogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def plot_grad_history(epochs, grad_history, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, grad_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean LoRA Gradient Norm")
    plt.title("Gradient Update Graph on LoRA Weights")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_weights_dir", type=str, default="weights/q1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="assignment5-q1")
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.save_weights_dir)
    dump_config(str(Path(args.output_dir) / "config.json"), args)

    device = get_device()
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_vit_small_for_cifar100()
    for p in model.parameters():
        p.requires_grad = False

    model = apply_lora_to_vit(
        model=model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = create_grad_scaler(args.use_amp)

    run_name = f"q1-lora-r{args.rank}-a{args.alpha}-d{args.dropout}"
    run = maybe_init_wandb(args, args.wandb_project, run_name)

    rows = []
    grad_hist = []
    best_val = -1.0
    best_path = Path(args.save_weights_dir) / f"best_lora_r{args.rank}_a{args.alpha}_d{args.dropout}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, grad_mean = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True, scaler=scaler
        )
        va_loss, va_acc, _ = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False, scaler=None
        )

        grad_hist.append(grad_mean)
        row = {
            "epoch": epoch,
            "training_loss": tr_loss,
            "validation_loss": va_loss,
            "training_accuracy": tr_acc,
            "validation_accuracy": va_acc,
            "lora_grad_norm": grad_mean,
        }
        rows.append(row)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_acc={va_acc:.4f}")

        if run is not None:
            import wandb

            wandb.log(row)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)

    train_table = pd.DataFrame(rows)
    train_table.to_csv(Path(args.output_dir) / "train_val_table.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_logits, test_labels = eval_logits(model, test_loader, device)
    test_acc = accuracy_from_logits(test_logits, test_labels)
    per_class_acc, class_counts = compute_per_class_accuracy(
        test_logits, test_labels, num_classes=100
    )

    hist_path = Path(args.output_dir) / "classwise_test_accuracy_hist.png"
    grad_path = Path(args.output_dir) / "lora_gradient_updates.png"
    plot_per_class_histogram(per_class_acc, str(hist_path))
    plot_grad_history(
        list(range(1, args.epochs + 1)),
        grad_hist,
        str(grad_path),
    )

    summary = {
        "lora_layers": "with",
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "overall_test_accuracy": test_acc,
        "trainable_parameters": count_trainable_params(model),
        "total_parameters": count_total_params(model),
        "best_validation_accuracy": best_val,
        "best_weights": str(best_path),
        "class_counts": class_counts.tolist(),
        "mean_classwise_accuracy": float(per_class_acc.mean()),
    }

    pd.DataFrame([summary]).to_csv(Path(args.output_dir) / "test_summary.csv", index=False)
    save_json(str(Path(args.output_dir) / "summary.json"), summary)

    if run is not None:
        import wandb

        wandb.log(
            {
                "test_accuracy": test_acc,
                "classwise_histogram": wandb.Image(str(hist_path)),
                "lora_gradient_updates": wandb.Image(str(grad_path)),
            }
        )
        run.finish()

    # Final cleanup
    cleanup_gpu()
    print("Training complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
