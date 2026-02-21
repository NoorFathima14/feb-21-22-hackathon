import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

# Allow running from project root
sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.datasets import get_dataloaders
from src.data.transforms import get_test_transforms
from src.data.explore import get_dataset_paths


def load_model(model_path, device):

    model = SpatialBaselineCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Adversarial Model Loaded Successfully")
    return model

def evaluate(model, test_loader, device):

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Adversarial Model - Confusion Matrix")
    plt.show()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir, test_dir = get_dataset_paths()

    test_transform = get_test_transforms()

    _, test_loader = get_dataloaders(
        train_dir,
        test_dir,
        train_transform=None,   # not needed
        test_transform=test_transform,
        batch_size=64,
        num_workers=2,
        use_subset=False
    )

    model_path = os.path.join("models", "adversarial_model.pth")

    model = load_model(model_path, device)

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()