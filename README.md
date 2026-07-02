# Handwritten Digit Recognition using CNN (PyTorch)

## Overview

This project implements a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** to recognize handwritten digits from the **MNIST dataset**.

The project demonstrates the complete deep learning workflow, including:

- Data preprocessing
- CNN architecture design
- Model training
- Model evaluation
- Model saving/loading
- Interactive digit prediction using a GUI

An interactive Tkinter application allows users to draw handwritten digits and classify them using the trained CNN.

## Interactive GUI

![GUI](outputs/sample_prediction_output.png)

---

## Features

- CNN built entirely from scratch using PyTorch
- Trained on the MNIST handwritten digit dataset
- Automatic GPU support (CUDA) if available
- Training loss visualization
- Model evaluation on the test dataset
- Confusion matrix generation
- Random prediction visualization
- Interactive digit drawing application
- Clean and modular project structure

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-Learn
- Pillow
- Tkinter

---

## CNN Architecture

```
Input Image (1 × 28 × 28)
        │
        ▼
Conv2D (32 Filters, 3×3)
        │
        ▼
ReLU
        │
        ▼
Conv2D (64 Filters, 3×3)
        │
        ▼
ReLU
        │
        ▼
Max Pooling (2×2)
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layer (128)
        │
        ▼
ReLU
        │
        ▼
Output Layer (10 Classes)
```

---

## Project Structure

```text
mnist-digit-classifier-pytorch/
│
├── data/
│
├── models/
│   └── cnn_model.pth
│
├── outputs/
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── predictions.png
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── digit_gui.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Jeshwanthg/mnist-digit-classifier-pytorch.git

cd mnist-digit-classifier-pytorch
```

Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run

```bash
cd src

python train.py
```

The script will

- Download the MNIST dataset
- Train the CNN
- Save the trained model
- Save the training loss curve

The trained model is stored in

```text
models/cnn_model.pth
```

---

## Evaluating the Model

Run

```bash
cd src

python evaluate.py
```

This generates

- Test Accuracy
- Confusion Matrix
- Random Prediction Examples

Results are saved inside

```text
outputs/
```

---

## Interactive Digit Recognition

Launch the GUI

```bash
cd src

python digit_gui.py
```

Features

- Draw a handwritten digit
- Click **Predict**
- View predicted digit
- View prediction confidence
- Clear the canvas and try again

---

## Example Outputs

### Training Loss

```
outputs/training_loss.png
```

---

### Confusion Matrix

```
outputs/confusion_matrix.png
```

---

### Prediction Examples

```
outputs/predictions.png
```

---

## Deep Learning Pipeline

```
MNIST Images
      │
      ▼
Image Transformations
      │
      ▼
DataLoader
      │
      ▼
CNN
      │
      ▼
Prediction
      │
      ▼
Loss Function
      │
      ▼
Backpropagation
      │
      ▼
Optimizer
      │
      ▼
Updated Weights
```

---

## Applications

- Handwritten Digit Recognition
- OCR Systems
- Postal Mail Sorting
- Bank Cheque Processing
- Form Digitization
- Deep Learning Education

---

## Future Improvements

- CNN Accuracy and Loss Curves
- TensorBoard Integration
- Webcam Digit Recognition
- Support for EMNIST Characters
- Larger CNN Architectures
- Custom Handwriting Dataset
- Export Model to ONNX
- Deploy using Streamlit or Gradio

---

## Author

**Jeshwanth Ganesh**

Master's Student – Autonomy Technologies

Computer Vision | Machine Learning | Robotics
