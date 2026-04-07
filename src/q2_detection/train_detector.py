import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from art.attacks.evasion import BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
from src.q2_attacks.models import resnet18_cifar10, resnet34_binary



def to_art_classifier(model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier



def generate_mixed_dataset(source_model, loader, device, attack_name: str, eps: float):
    source_model.eval()
    classifier = to_art_classifier(source_model, device)

    if attack_name == "pgd":
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps / 4, max_iter=10)
    elif attack_name == "bim":
        attack = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=eps / 5, max_iter=10)
    else:
        raise ValueError("attack_name must be pgd or bim")

    clean_batches = []
    adv_batches = []

    for x, _ in tqdm(loader, leave=False):
        x = x.to(device)
        x_adv_np = attack.generate(x=x.detach().cpu().numpy())
        x_adv = torch.tensor(x_adv_np, dtype=x.dtype)
        clean_batches.append(x.cpu())
        adv_batches.append(x_adv)

    clean = torch.cat(clean_batches, dim=0)
    adv = torch.cat(adv_batches, dim=0)

    x_all = torch.cat([clean, adv], dim=0)
    y_all = torch.cat(
        [
            torch.zeros(clean.size(0), dtype=torch.long),
            torch.ones(adv.size(0), dtype=torch.long),
        ],
        dim=0,
    )

    perm = torch.randperm(x_all.size(0))
    x_all = x_all[perm]
    y_all = y_all[perm]
    return x_all, y_all, clean[:10], adv[:10]



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



def log_samples_wandb(run, clean_10, adv_10, attack_name):
    import wandb

    imgs = []
    for i in range(min(10, clean_10.size(0))):
        c = np.transpose(clean_10[i].numpy(), (1, 2, 0))
        a = np.transpose(adv_10[i].numpy(), (1, 2, 0))
        imgs.append(wandb.Image(c, caption=f"{attack_name}-clean-{i}"))
        imgs.append(wandb.Image(a, caption=f"{attack_name}-adv-{i}"))
    wandb.log({f"samples_{attack_name}": imgs})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--source_weights", type=str, default="weights/q2/resnet18_clean_best.pt")
    parser.add_argument("--attack", type=str, choices=["pgd", "bim"], required=True)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/q2/detectors")
    parser.add_argument("--weights_dir", type=str, default="weights/q2/detectors")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="assignment5-q2")
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.weights_dir)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training {args.attack.upper()} detector...")
    
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    source_model = resnet18_cifar10().to(device)
    source_model.load_state_dict(torch.load(args.source_weights, map_location=device, weights_only=True))

    print("Generating adversarial examples for training set...")
    x_tr, y_tr, clean_10, adv_10 = generate_mixed_dataset(source_model, train_loader, device, args.attack, args.eps)
    print("Generating adversarial examples for validation set...")
    x_va, y_va, _, _ = generate_mixed_dataset(source_model, val_loader, device, args.attack, args.eps)
    print("Generating adversarial examples for test set...")
    x_te, y_te, _, _ = generate_mixed_dataset(source_model, test_loader, device, args.attack, args.eps)

    det_train = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    det_val = DataLoader(TensorDataset(x_va, y_va), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    det_test = DataLoader(TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    detector = resnet34_binary().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr)
    
    # Mixed precision training
    scaler = create_grad_scaler(args.use_amp)

    run_name = f"q2-detector-{args.attack}"
    run = maybe_init_wandb(args, args.wandb_project, run_name)

    rows = []
    best_val = -1.0
    best_path = Path(args.weights_dir) / f"resnet34_detector_{args.attack}.pt"

    print("Training detector...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(detector, det_train, optimizer, criterion, device, train=True, scaler=scaler)
        va_loss, va_acc = run_epoch(detector, det_val, optimizer, criterion, device, train=False, scaler=None)

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
            torch.save(detector.state_dict(), best_path)

    detector.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_loss, te_acc = run_epoch(detector, det_test, optimizer, criterion, device, train=False, scaler=None)

    table = pd.DataFrame(rows)
    out_dir = Path(args.output_dir) / args.attack
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "train_val_table.csv", index=False)

    summary = {
        "attack": args.attack,
        "eps": args.eps,
        "best_validation_accuracy": best_val,
        "test_detection_accuracy": te_acc,
        "test_detection_loss": te_loss,
        "best_weights": str(best_path),
        "target_requirement_met": te_acc >= 0.70,
    }
    save_json(str(out_dir / "summary.json"), summary)

    print(f"\n{args.attack.upper()} Detector Results:")
    print(f"  Test detection accuracy: {te_acc:.4f} (target: >=70%)")
    if te_acc >= 0.70:
        print("  ✓ Target accuracy achieved!")
    else:
        print("  ✗ Target accuracy not achieved.")

    if run is not None:
        log_samples_wandb(run, clean_10, adv_10, args.attack)
        import wandb

        wandb.log(
            {
                "test_detection_accuracy": te_acc,
                "test_detection_loss": te_loss,
                "target_requirement_met": float(te_acc >= 0.70),
            }
        )
        run.finish()

    # Final cleanup
    cleanup_gpu()
    print("Training complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
