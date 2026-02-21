import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.explore import get_dataset_paths


def get_test_loader(batch_size=64):

    _, test_dir = get_dataset_paths()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def get_test_dataset():

    _, test_dir = get_dataset_paths()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    return test_dataset