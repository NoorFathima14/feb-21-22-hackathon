import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

def get_dataloaders(train_dir, test_dir,
                    train_transform,
                    test_transform,
                    batch_size=64,
                    num_workers=2,
                    use_subset=False,
                    subset_size=20000):

    train_dataset = ImageFolder(root=train_dir,
                                transform=train_transform)

    test_dataset = ImageFolder(root=test_dir,
                               transform=test_transform)

    if use_subset:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader