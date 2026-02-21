import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.utils.data_utils import get_test_loader
from src.attacks.evaluate_fgsm import evaluate_adversarial


def load_model(path, device):

    model = SpatialBaselineCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate_clean(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = get_test_loader()

    baseline_path = "models/baseline_model.pth"
    adversarial_path = "models/adversarial_model.pth"

    baseline_model = load_model(baseline_path, device)
    robust_model = load_model(adversarial_path, device)

    baseline_clean_acc = evaluate_clean(baseline_model, test_loader, device)
    robust_clean_acc = evaluate_clean(robust_model, test_loader, device)

    print("\nClean Accuracy")
    print(f"Baseline: {baseline_clean_acc:.4f}")
    print(f"Robust:   {robust_clean_acc:.4f}")

    epsilons = [0.01, 0.03, 0.05, 0.07, 0.1]

    print("\nAdversarial Accuracy Comparison")

    for eps in epsilons:

        acc_base = evaluate_adversarial(baseline_model, test_loader, eps, device)
        acc_rob  = evaluate_adversarial(robust_model, test_loader, eps, device)

        print(f"\nEpsilon: {eps}")
        print(f"Baseline: {acc_base:.4f}")
        print(f"Robust:   {acc_rob:.4f}")


if __name__ == "__main__":
    main()