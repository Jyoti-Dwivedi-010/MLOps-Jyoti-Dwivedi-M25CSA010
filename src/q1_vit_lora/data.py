from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def get_cifar100_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=16),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=train_tfms,
        download=True,
    )
    val_set = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=eval_tfms,
        download=False,
    )

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_subset, val_subset = torch_random_split(full_train, val_set, train_size, val_size)

    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=eval_tfms,
        download=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader



def torch_random_split(train_set_aug, train_set_eval, train_size, val_size):
    import torch

    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(train_set_aug)),
        [train_size, val_size],
        generator=gen,
    )

    train_subset = torch.utils.data.Subset(train_set_aug, train_idx.indices)
    val_subset = torch.utils.data.Subset(train_set_eval, val_idx.indices)
    return train_subset, val_subset
