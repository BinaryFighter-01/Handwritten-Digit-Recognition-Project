# Handwritten Digit Recognition Project

## Overview
This project implements a handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application provides a simple GUI using Tkinter where users can draw digits on a canvas, and the system predicts the digit using a pre-trained model.

## Features
- Draw digits on a 200x200 canvas
- Predict handwritten digits (0-9) with confidence scores
- Clear canvas functionality
- Pre-trained CNN model using TensorFlow/Keras
- Automatically trains and saves model if not already present

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Pillow (PIL)
- NumPy
- Tkinter (usually included with Python)

## Installation
1. Clone or download this repository.
2. Install the required packages:
```bash
pip install tensorflow pillow numpy
```

## Usage
1. Run the application:
```bash
python digit_recognition.py
```
2. The GUI will open with a white canvas.
3. Draw a digit (0-9) using your mouse (hold left-click to draw).
4. Click "Predict Digit" to see the model's prediction and confidence score.
5. Click "Clear Canvas" to reset and draw another digit.

## Project Structure
- `digit_recognition.py`: Main application script containing the GUI and model logic.
- `mnist_model.h5`: Pre-trained model file (generated after first run).

## How It Works
- The model is a CNN with two convolutional layers, max-pooling, and dense layers.
- If `mnist_model.h5` exists, it loads the pre-trained model; otherwise, it trains a new model on the MNIST dataset and saves it.
- Drawings are resized to 28x28 pixels (MNIST format) before prediction.
- Predictions include the recognized digit and confidence score.

## Notes
- The model is trained for 5 epochs by default. Adjust the `epochs` parameter in `load_or_train_model()` for better accuracy if needed.
- Ensure sufficient disk space for saving the model file (~1MB).
- The GUI is minimal and optimized for simplicity.

## License
MIT License