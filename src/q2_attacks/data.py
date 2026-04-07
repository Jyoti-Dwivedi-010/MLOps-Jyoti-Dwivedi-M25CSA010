from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms



def get_cifar10_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose([transforms.ToTensor()])

    full_train_aug = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_tfms,
        download=True,
    )
    full_train_eval = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=eval_tfms,
        download=False,
    )

    train_size = int(0.9 * len(full_train_aug))
    val_size = len(full_train_aug) - train_size

    gen = torch_gen()
    train_idx, val_idx = random_split(
        range(len(full_train_aug)),
        [train_size, val_size],
        generator=gen,
    )

    train_set = subset(full_train_aug, train_idx.indices)
    val_set = subset(full_train_eval, val_idx.indices)

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=eval_tfms,
        download=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
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



def subset(dataset, indices):
    from torch.utils.data import Subset

    return Subset(dataset, indices)



def torch_gen():
    import torch

    return torch.Generator().manual_seed(42)
