import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.explore import get_dataset_paths
from src.explainability.gradcam import GradCAM
from src.attacks.fgsm import fgsm_attack, predict_with_confidence

# Load Model
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


def find_high_conf_fake(model, dataset, device, threshold=0.9):

    for i in range(len(dataset)):
        image, label = dataset[i]

        if label == 0:  # FAKE
            pred, conf = predict_with_confidence(model, image, device)

            if pred == 0 and conf > threshold:
                print("Found high-confidence FAKE example")
                print("Confidence:", conf)
                return i

    return None


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

    fake_index = find_high_conf_fake(model, test_dataset, device)

    if fake_index is None:
        print("No suitable FAKE image found.")
        return

    image, label = test_dataset[fake_index]

    epsilons = [0.01, 0.03, 0.05, 0.07, 0.1]

    # Visualize adversarial images
    plt.figure(figsize=(15, 5))

    for i, eps in enumerate(epsilons):

        adv_image = fgsm_attack(model, image, label, eps, device)
        pred, conf = predict_with_confidence(model, adv_image, device)

        img = adv_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5) + 0.5

        plt.subplot(1, len(epsilons), i + 1)
        plt.imshow(img)
        plt.title(f"ε={eps}\nPred:{pred}\nConf:{conf:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # GradCAM Comparison
    gradcam = GradCAM(model, model.conv3)

    adv_image = fgsm_attack(model, image, label, epsilon=0.1, device=device)

    orig_cam = gradcam.generate(image.unsqueeze(0).to(device))
    adv_cam = gradcam.generate(adv_image.unsqueeze(0).to(device))

    orig_img = image.permute(1, 2, 0).numpy()
    orig_img = (orig_img * 0.5) + 0.5

    adv_img = adv_image.permute(1, 2, 0).cpu().numpy()
    adv_img = (adv_img * 0.5) + 0.5

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(orig_img)
    plt.imshow(orig_cam, cmap="jet", alpha=0.5)
    plt.title("Original Grad-CAM")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(adv_img)
    plt.title("Adversarial Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(adv_img)
    plt.imshow(adv_cam, cmap="jet", alpha=0.5)
    plt.title("Adversarial Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()