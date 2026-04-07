import atexit
import gc
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_index: Optional[int] = None) -> torch.device:
    """Get the CUDA device. Uses CUDA_VISIBLE_DEVICES if set, otherwise gpu_index."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # CUDA_VISIBLE_DEVICES remaps GPU indices, so use cuda:0 when it's set
    # The actual GPU selection is handled by CUDA_VISIBLE_DEVICES environment variable
    device = torch.device("cuda:0")
    
    # Enable cuDNN benchmark for faster training with fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    return device


def cleanup_gpu() -> None:
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def register_cleanup_on_exit() -> None:
    """Register GPU cleanup to run when the program exits."""
    atexit.register(cleanup_gpu)



def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()



def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def save_json(path: str, payload: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def dump_config(path: str, cfg) -> None:
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif isinstance(cfg, dict):
        payload = cfg
    else:
        payload = cfg.__dict__
    save_json(path, payload)



def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def count_total_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())



def compute_per_class_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for p, t in zip(preds, labels_np):
        total[t] += 1
        if p == t:
            correct[t] += 1

    acc = np.divide(correct, np.maximum(total, 1), dtype=np.float64)
    return acc, total



def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += float(torch.sum(g * g).item())
    return total_sq ** 0.5



def maybe_init_wandb(args, project: str, run_name: str):
    if not getattr(args, "use_wandb", False):
        return None
    import wandb

    config = vars(args).copy()
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def get_amp_context(enabled: bool = True):
    """Get autocast context manager for mixed precision training."""
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    else:
        import contextlib
        return contextlib.nullcontext()


def create_grad_scaler(enabled: bool = True):
    """Create gradient scaler for mixed precision training."""
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None
