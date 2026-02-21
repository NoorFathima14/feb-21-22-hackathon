import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.datasets import get_dataloaders
from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.explore import get_dataset_paths

# Adversarial Training Step
def train_one_epoch_adv(model, loader, criterion, optimizer, device, epsilon):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        # FGSM
        adv_images = images + epsilon * images.grad.sign()
        adv_images = torch.clamp(adv_images, -1, 1).detach()

        optimizer.zero_grad()

        outputs_clean = model(images.detach())
        loss_clean = criterion(outputs_clean, labels)

        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)

        total_loss = (loss_clean + loss_adv) / 2
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        _, predicted = torch.max(outputs_adv, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total


# --------------------------------------
# Clean Evaluation (reuse logic)
# --------------------------------------
def evaluate(model, loader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir, test_dir = get_dataset_paths()

    train_loader, test_loader = get_dataloaders(
        train_dir,
        test_dir,
        get_train_transforms(),
        get_test_transforms(),
        batch_size=64,
        num_workers=2,
        use_subset=False
    )

    model = SpatialBaselineCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epsilon_train = 0.03
    num_epochs = 5

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch_adv(
            model, train_loader, criterion, optimizer, device, epsilon_train
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/adversarial_model.pth")

    print("Saved as models/adversarial_model.pth")


if __name__ == "__main__":
    main()