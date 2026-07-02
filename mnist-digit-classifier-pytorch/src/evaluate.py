"""
evaluate.py

Evaluate the trained CNN model on the MNIST test dataset.

This script:
1. Loads the trained model
2. Evaluates accuracy
3. Creates a confusion matrix
4. Saves prediction examples
"""

import os
import random
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from model import CNN

# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("../outputs", exist_ok=True)

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================================
# Test Dataset
# ==========================================================

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ==========================================================
# Load Model
# ==========================================================

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "../models/cnn_model.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# Evaluation
# ==========================================================

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

accuracy = 100 * correct / total

print(f"\nTest Accuracy : {accuracy:.2f}%")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(10,8))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues
)

plt.title("Confusion Matrix")

plt.colorbar()

classes = list(range(10))

tick_marks = np.arange(len(classes))

plt.xticks(
    tick_marks,
    classes
)

plt.yticks(
    tick_marks,
    classes
)

threshold = cm.max() / 2

for i, j in itertools.product(
    range(cm.shape[0]),
    range(cm.shape[1])
):

    plt.text(
        j,
        i,
        format(cm[i, j], "d"),
        horizontalalignment="center",
        color="white" if cm[i, j] > threshold else "black"
    )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.tight_layout()

plt.savefig(
    "../outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved.")

# ==========================================================
# Random Predictions
# ==========================================================

fig = plt.figure(figsize=(10,10))

indices = random.sample(
    range(len(test_dataset)),
    9
)

for i, idx in enumerate(indices):

    image, label = test_dataset[idx]

    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_input)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    plt.subplot(3,3,i+1)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"P:{prediction}  T:{label}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "../outputs/predictions.png",
    dpi=300
)

plt.close()

print("Prediction Examples Saved.")