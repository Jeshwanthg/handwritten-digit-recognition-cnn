"""
digit_gui.py

Interactive Handwritten Digit Recognition GUI using Tkinter.

Run:
    python digit_gui.py
"""

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CNN

# ==========================================
# Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Model
# ==========================================

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "../models/cnn_model.pth",
        map_location=device
    )
)

model.eval()

# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# GUI
# ==========================================

WIDTH = 280
HEIGHT = 280
BRUSH_SIZE = 14

root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.resizable(False, False)

prediction_var = tk.StringVar(value="-")
confidence_var = tk.StringVar(value="-")

canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="black",
    cursor="cross"
)

canvas.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

# PIL image used for prediction
image = Image.new("L", (WIDTH, HEIGHT), color=0)
draw = ImageDraw.Draw(image)


def paint(event):
    x = event.x
    y = event.y

    canvas.create_oval(
        x - BRUSH_SIZE,
        y - BRUSH_SIZE,
        x + BRUSH_SIZE,
        y + BRUSH_SIZE,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (
            x - BRUSH_SIZE,
            y - BRUSH_SIZE,
            x + BRUSH_SIZE,
            y + BRUSH_SIZE
        ),
        fill=255
    )


canvas.bind("<B1-Motion>", paint)


def clear_canvas():

    global image, draw

    canvas.delete("all")

    image = Image.new("L", (WIDTH, HEIGHT), color=0)
    draw = ImageDraw.Draw(image)

    prediction_var.set("-")
    confidence_var.set("-")


def predict():

    img = image.copy()

    # Crop empty space
    bbox = img.getbbox()

    if bbox is None:
        return

    img = img.crop(bbox)

    # Make square
    size = max(img.size)

    square = Image.new("L", (size, size), 0)

    square.paste(
        img,
        (
            (size - img.size[0]) // 2,
            (size - img.size[1]) // 2
        )
    )

    img = transform(square)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        probs = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probs,
            dim=1
        )

    prediction_var.set(str(prediction.item()))

    confidence_var.set(
        f"{confidence.item()*100:.2f}%"
    )


# ==========================================
# Buttons
# ==========================================

predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    width=15,
    height=2,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 11, "bold")
)

predict_button.grid(
    row=1,
    column=0,
    padx=10,
    pady=10
)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    width=15,
    height=2,
    bg="#f44336",
    fg="white",
    font=("Arial", 11, "bold")
)

clear_button.grid(
    row=1,
    column=1,
    padx=10,
    pady=10
)

# ==========================================
# Prediction Labels
# ==========================================

tk.Label(
    root,
    text="Prediction",
    font=("Arial", 14, "bold")
).grid(row=2, column=0, pady=5)

tk.Label(
    root,
    textvariable=prediction_var,
    font=("Arial", 28, "bold"),
    fg="blue"
).grid(row=3, column=0)

tk.Label(
    root,
    text="Confidence",
    font=("Arial", 14, "bold")
).grid(row=2, column=1)

tk.Label(
    root,
    textvariable=confidence_var,
    font=("Arial", 24, "bold"),
    fg="green"
).grid(row=3, column=1)

root.mainloop()