"""
[Optional] Partial Freeze Experiment:
- Keep some ViT layers trainable (last N blocks)
- Apply LoRA only to frozen layers
- Compare with fully frozen + LoRA approach
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
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
from src.q1_vit_lora.modeling import create_vit_small_for_cifar100


class PartialFreezeLoRAViT(nn.Module):
    """
    ViT with partial freezing and LoRA:
    - Last `trainable_blocks` transformer blocks are kept trainable (no LoRA)
    - Earlier blocks are frozen and have LoRA applied
    - Classification head is always trainable
    """

    def __init__(self, base_model, peft_model, trainable_blocks):
        super().__init__()
        self.peft_model = peft_model
        self.trainable_blocks = trainable_blocks

    def forward(self, x):
        return self.peft_model.base_model.model(x)


def apply_partial_freeze_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    trainable_blocks: int = 2,
):
    """
    Apply LoRA only to the frozen blocks of the ViT model.
    Keep the last `trainable_blocks` blocks trainable (no LoRA).
    
    ViT-S has 12 transformer blocks (blocks.0 to blocks.11).
    If trainable_blocks=2, blocks.10 and blocks.11 are trainable,
    and blocks.0-9 are frozen with LoRA.
    """
    num_total_blocks = 12  # ViT-S has 12 blocks
    frozen_block_indices = list(range(num_total_blocks - trainable_blocks))
    trainable_block_indices = list(range(num_total_blocks - trainable_blocks, num_total_blocks))

    # First freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classification head
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True

    # Unfreeze last N transformer blocks
    for block_idx in trainable_block_indices:
        block = model.blocks[block_idx]
        for p in block.parameters():
            p.requires_grad = True

    # Build target_modules for LoRA - only frozen blocks
    target_modules = []
    for block_idx in frozen_block_indices:
        target_modules.extend([
            f"blocks.{block_idx}.attn.qkv",
        ])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    
    # Re-ensure trainable blocks stay trainable after PEFT wrapping
    for block_idx in trainable_block_indices:
        block_name = f"blocks.{block_idx}"
        for name, param in peft_model.named_parameters():
            if block_name in name and "lora_" not in name:
                param.requires_grad = True

    # Ensure head is trainable
    for name, param in peft_model.named_parameters():
        if "head" in name:
            param.requires_grad = True

    return PartialFreezeLoRAViT(model, peft_model, trainable_blocks)


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
            if lora_params:
                grad_values.append(grad_norm(lora_params))

        bs = images.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy_from_logits(logits, labels) * bs

    grad_mean = float(sum(grad_values) / max(len(grad_values), 1)) if grad_values else 0.0
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
    plt.title("Class-wise Test Accuracy Histogram (Partial Freeze + LoRA)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grad_history(epochs, grad_history, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, grad_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean LoRA Gradient Norm")
    plt.title("Gradient Update Graph on LoRA Weights (Partial Freeze)")
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
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (best from Optuna)")
    parser.add_argument("--alpha", type=int, default=4, help="LoRA alpha (best from Optuna)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trainable_blocks", type=int, default=2, 
                        help="Number of last transformer blocks to keep trainable (no LoRA)")
    parser.add_argument("--output_dir", type=str, default="results/q1/partial_freeze")
    parser.add_argument("--save_weights_dir", type=str, default="weights/q1/partial_freeze")
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
    print(f"Partial Freeze Experiment: {args.trainable_blocks} blocks trainable, "
          f"rest frozen with LoRA (r={args.rank}, α={args.alpha})")

    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_vit_small_for_cifar100()
    model = apply_partial_freeze_lora(
        model=model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        trainable_blocks=args.trainable_blocks,
    ).to(device)

    # Count params
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision training
    scaler = create_grad_scaler(args.use_amp)

    run_name = f"q1-partial-freeze-b{args.trainable_blocks}-r{args.rank}-a{args.alpha}"
    run = maybe_init_wandb(args, args.wandb_project, run_name)

    rows = []
    grad_hist = []
    best_val = -1.0
    best_path = Path(args.save_weights_dir) / f"best_partial_freeze_b{args.trainable_blocks}_r{args.rank}_a{args.alpha}.pt"

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
        "experiment_type": "partial_freeze_lora",
        "trainable_blocks": args.trainable_blocks,
        "frozen_blocks_with_lora": 12 - args.trainable_blocks,
        "lora_layers": "with (on frozen blocks only)",
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "overall_test_accuracy": test_acc,
        "trainable_parameters": trainable,
        "total_parameters": total,
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
    print(f"\nPartial Freeze Experiment Complete!")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val:.4f}")
    print(f"Trainable Parameters: {trainable:,}")
    print("GPU memory cleaned up.")


if __name__ == "__main__":
    main()
