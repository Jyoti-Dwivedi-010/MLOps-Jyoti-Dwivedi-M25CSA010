import argparse

import numpy as np
import torch
from art.attacks.evasion import BasicIterativeMethod, FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from torch import nn

from src.common import cleanup_gpu, get_device, register_cleanup_on_exit
from src.q2_attacks.data import get_cifar10_dataloaders
from src.q2_attacks.fgsm_compare import fgsm_scratch
from src.q2_attacks.models import resnet18_cifar10



def to_art_classifier(model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )



def log_pair_images(prefix, clean, adv):
    import wandb

    images = []
    n = min(10, clean.size(0))
    for i in range(n):
        c = np.transpose(clean[i].numpy(), (1, 2, 0))
        a = np.transpose(adv[i].numpy(), (1, 2, 0))
        images.append(wandb.Image(c, caption=f"{prefix}-clean-{i}"))
        images.append(wandb.Image(a, caption=f"{prefix}-adv-{i}"))
    wandb.log({f"samples_{prefix}": images})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--weights_path", type=str, default="weights/q2/resnet18_clean_best.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--wandb_project", type=str, default="assignment5-q2")
    parser.add_argument("--run_name", type=str, default="q2-attack-samples")
    args = parser.parse_args()

    # Register GPU cleanup on exit
    register_cleanup_on_exit()

    import wandb

    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    device = get_device()
    print(f"Using device: {device}")
    
    _, _, test_loader = get_cifar10_dataloaders(args.data_root, args.batch_size, args.num_workers)

    x, y = next(iter(test_loader))
    x = x.to(device)
    y = y.to(device)

    model = resnet18_cifar10().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model.eval()

    classifier = to_art_classifier(model, device)

    print("Generating attack samples...")
    x_fgsm_scratch = fgsm_scratch(model, x, y, args.eps).detach().cpu()
    x_fgsm_art = torch.tensor(
        FastGradientMethod(estimator=classifier, eps=args.eps).generate(x=x.detach().cpu().numpy())
    )
    x_pgd = torch.tensor(
        ProjectedGradientDescent(estimator=classifier, eps=args.eps, eps_step=args.eps / 4, max_iter=10).generate(
            x=x.detach().cpu().numpy()
        )
    )
    x_bim = torch.tensor(
        BasicIterativeMethod(estimator=classifier, eps=args.eps, eps_step=args.eps / 5, max_iter=10).generate(
            x=x.detach().cpu().numpy()
        )
    )

    clean = x.detach().cpu()
    print("Logging samples to WandB...")
    log_pair_images("fgsm_scratch", clean, x_fgsm_scratch)
    log_pair_images("fgsm_art", clean, x_fgsm_art)
    log_pair_images("pgd", clean, x_pgd)
    log_pair_images("bim", clean, x_bim)

    run.finish()
    
    # Final cleanup
    cleanup_gpu()
    print("Attack sample logging complete. GPU memory cleaned up.")



if __name__ == "__main__":
    main()
