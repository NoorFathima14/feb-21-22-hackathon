import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.explore import get_dataset_paths


# Load Model
def load_model(model_path, device):

    model = SpatialBaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model Loaded Successfully")
    return model


# Get Test Loader
def get_test_loader(test_dir, batch_size=64):

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


# Evaluate
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
    plt.title("Confusion Matrix")
    plt.show()


# Main
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir, test_dir = get_dataset_paths()

    # ---- MODEL PATH HANDLING ----
    # Local path (your case)
    local_model_path = os.path.join("models", "baseline_model.pth")

    # Kaggle fallback
    kaggle_model_path = "/kaggle/working/baseline_model.pth"

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print("Running locally.")
    else:
        model_path = kaggle_model_path
        print("Running in Kaggle.")

    model = load_model(model_path, device)

    test_loader = get_test_loader(test_dir)

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()