import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

sys.path.append(os.path.abspath("."))

from src.utils.model_utils import load_baseline_model
from src.utils.data_utils import get_test_loader


def evaluate_adversarial(model, dataloader, epsilon, device):

    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        perturbed_images = images + epsilon * images.grad.sign()
        perturbed_images = torch.clamp(perturbed_images, -1, 1)

        outputs_adv = model(perturbed_images)
        _, preds = torch.max(outputs_adv, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_baseline_model(device)
    test_loader = get_test_loader()

    epsilons = [0.01, 0.03, 0.05, 0.07, 0.1]
    adv_accuracies = []

    for eps in epsilons:

        acc = evaluate_adversarial(
            model,
            test_loader,
            epsilon=eps,
            device=device
        )

        print(f"Epsilon: {eps} | Adversarial Accuracy: {acc:.4f}")
        adv_accuracies.append(acc)

    plt.plot(epsilons, adv_accuracies, marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Adversarial Robustness Curve")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()