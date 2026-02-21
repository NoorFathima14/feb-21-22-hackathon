import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpatialBaselineCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # After 3 poolings: 32x32 → 16x16 → 8x8 → 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x