import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.spatial_baseline import SpatialBaselineCNN
from src.data.datasets import get_dataloaders
from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.explore import get_dataset_paths

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


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

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dir, test_dir = get_dataset_paths()

    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    train_loader, test_loader = get_dataloaders(
        train_dir,
        test_dir,
        train_transform,
        test_transform,
        batch_size=64,
        num_workers=2,
        use_subset=False
    )

    model = SpatialBaselineCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "baseline_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()