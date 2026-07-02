"""
train.py

Train a Convolutional Neural Network (CNN) on the MNIST dataset.

This script:
1. Loads and preprocesses the MNIST dataset
2. Creates DataLoaders
3. Initializes the CNN model
4. Trains the network
5. Saves the trained model
6. Saves the training loss curve
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import CNN

# ==========================================================
# Device Configuration
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")

# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# ==========================================================
# Create Output Directories
# ==========================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# ==========================================================
# Image Transformations
# ==========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================================
# Load MNIST Dataset
# ==========================================================

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Training Images : {len(train_dataset)}")

# ==========================================================
# Initialize CNN
# ==========================================================

model = CNN().to(device)

# ==========================================================
# Loss Function & Optimizer
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================================
# Lists for Plotting
# ==========================================================

train_losses = []

# ==========================================================
# Training Loop
# ==========================================================

print("\nTraining Started...\n")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Move data to GPU (if available)
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)

        # Compute Loss
        loss = criterion(outputs, labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f}"
    )

print("\nTraining Finished!")

# ==========================================================
# Save Model
# ==========================================================

MODEL_PATH = "../models/cnn_model.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(f"\nModel saved to:\n{MODEL_PATH}")

# ==========================================================
# Plot Training Loss
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    marker="o",
    linewidth=2
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

LOSS_PLOT = "../outputs/training_loss.png"

plt.savefig(
    LOSS_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Training loss curve saved to:\n{LOSS_PLOT}")

print("\nTraining Complete!")