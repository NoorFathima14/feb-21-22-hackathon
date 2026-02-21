# ======================================
# src/explainability/visualize_gradcam.py
# ======================================

import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.explore import get_dataset_paths
from src.explainability.gradcam import GradCAM


def load_model(device):

    local_model_path = os.path.join("models", "baseline_model.pth")
    kaggle_model_path = "/kaggle/working/baseline_model.pth"

    if os.path.exists(local_model_path):
        model_path = local_model_path
        print("Running locally.")
    else:
        model_path = kaggle_model_path
        print("Running in Kaggle.")

    model = SpatialBaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, test_dir = get_dataset_paths()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    model = load_model(device)

    gradcam = GradCAM(model, model.conv3)

    image, label = test_dataset[0]
    input_tensor = image.unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor)

    # Unnormalize
    img = image.permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()