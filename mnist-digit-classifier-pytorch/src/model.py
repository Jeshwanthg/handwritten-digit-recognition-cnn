"""
model.py

Defines the Convolutional Neural Network (CNN) architecture
used for handwritten digit classification on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
        Input (1 x 28 x 28)
            ↓
        Conv2D (32 filters)
            ↓
        ReLU
            ↓
        Conv2D (64 filters)
            ↓
        ReLU
            ↓
        Max Pooling
            ↓
        Flatten
            ↓
        Fully Connected (128 neurons)
            ↓
        Output Layer (10 classes)
    """

    def __init__(self):
        super().__init__()

        # First Convolution Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        # Second Convolution Layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully Connected Layer
        self.fc1 = nn.Linear(
            64 * 12 * 12,
            128
        )

        # Output Layer
        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):
        """
        Forward propagation through the network.
        """

        x = F.relu(self.conv1(x))

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x