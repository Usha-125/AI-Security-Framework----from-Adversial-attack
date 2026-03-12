import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorCNN(nn.Module):

    def __init__(self):
        super(DetectorCNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)  # binary classification

    def forward(self, x):

        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x