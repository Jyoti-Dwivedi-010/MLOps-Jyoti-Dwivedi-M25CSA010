import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch import nn
from tqdm import tqdm

from src.common import cleanup_gpu, ensure_dir, get_device, maybe_init_wandb, register_cleanup_on_exit, save_json
from src.q2_attacks.data import get_cifar10_dataloaders
from src.q2_attacks.models import resnet18_cifar10



def fgsm_scratch(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    perturb = eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv + perturb, 0.0, 1.0).detach()
    return x_adv



def batch_accuracy(model, x, y):
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()



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



def make_panel(clean, adv_scratch, adv_art, out_file):
    n = min(10, clean.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        for j, img in enumerate([clean[i], adv_scratch[i], adv_art[i]]):
            axes[i, j].imshow(np.transpose(img, (1, 2, 0)))
            axes[i, j].axis("off")

    axes[0, 0].set_title("Clean")
    axes[0, 1].set_title("FGSM Scratch")
    axes[0, 2].set_title("FGSM ART")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="weights/q2/resnet18_clean_best.pt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--output_dir", type=str, default="results/q2/fgsm_compare")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="assignment5-q2")
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    ensure_dir(args.output_dir)
    device = get_device()
    print(f"Using device: {device}")
    
    _, _, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = resnet18_cifar10().to(device)
    state = torch.load(args.weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    art_classifier = to_art_classifier(model, device)
    art_attack = FastGradientMethod(estimator=art_classifier, eps=args.eps)

    run = maybe_init_wandb(args, args.wandb_project, "q2-fgsm-compare")

    clean_correct = 0
    scratch_correct = 0
    art_correct = 0
    total = 0

    panel_clean = None
    panel_scratch = None
    panel_art = None

    print("Running FGSM attack comparison...")
    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)

        x_scratch = fgsm_scratch(model, x, y, args.eps)
        x_art_np = art_attack.generate(x=x.detach().cpu().numpy())
        x_art = torch.tensor(x_art_np, dtype=x.dtype, device=device)

        with torch.no_grad():
            clean_pred = torch.argmax(model(x), dim=1)
            scratch_pred = torch.argmax(model(x_scratch), dim=1)
            art_pred = torch.argmax(model(x_art), dim=1)

        clean_correct += int((clean_pred == y).sum().item())
        scratch_correct += int((scratch_pred == y).sum().item())
        art_correct += int((art_pred == y).sum().item())
        total += x.size(0)

        if panel_clean is None:
            panel_clean = x[:10].detach().cpu().numpy()
            panel_scratch = x_scratch[:10].detach().cpu().numpy()
            panel_art = x_art[:10].detach().cpu().numpy()

    clean_acc = clean_correct / total
    scratch_acc = scratch_correct / total
    art_acc = art_correct / total

    panel_path = Path(args.output_dir) / "fgsm_clean_vs_scratch_vs_art.png"
    make_panel(panel_clean, panel_scratch, panel_art, str(panel_path))

    rows = [
        {"setting": "clean", "epsilon": 0.0, "accuracy": clean_acc},
        {"setting": "fgsm_scratch", "epsilon": args.eps, "accuracy": scratch_acc},
        {"setting": "fgsm_art", "epsilon": args.eps, "accuracy": art_acc},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(Path(args.output_dir) / "accuracy_comparison.csv", index=False)

    summary = {
        "clean_accuracy": clean_acc,
        "fgsm_scratch_accuracy": scratch_acc,
        "fgsm_art_accuracy": art_acc,
        "performance_drop_scratch": clean_acc - scratch_acc,
        "performance_drop_art": clean_acc - art_acc,
        "epsilon": args.eps,
    }
    save_json(str(Path(args.output_dir) / "summary.json"), summary)

    print(f"\nResults:")
    print(f"  Clean accuracy: {clean_acc:.4f}")
    print(f"  FGSM (scratch): {scratch_acc:.4f} (drop: {clean_acc - scratch_acc:.4f})")
    print(f"  FGSM (ART):     {art_acc:.4f} (drop: {clean_acc - art_acc:.4f})")

    if run is not None:
        import wandb

        wandb.log(
            {
                "clean_accuracy": clean_acc,
                "fgsm_scratch_accuracy": scratch_acc,
                "fgsm_art_accuracy": art_acc,
                "comparison_table": wandb.Table(dataframe=table),
                "fgsm_samples": wandb.Image(str(panel_path)),
            }
        )
        run.finish()

    # Final cleanup
    cleanup_gpu()
    print("FGSM comparison complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
