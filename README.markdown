Handwritten Digit Recognition Project
Overview
This project implements a handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application provides a Tkinter-based GUI where users can draw digits on a canvas, and the system predicts the digit using a pre-trained model.
Features

Draw digits on a 200x200 canvas with smooth mouse-based drawing
Predict handwritten digits (0-9) with confidence scores
Clear canvas functionality
Pre-trained CNN model using TensorFlow/Keras
Automatically trains and saves model if not already present
Error handling for model loading/training and predictions

Requirements

Python 3.6+
TensorFlow 2.x
Pillow (PIL)
NumPy
Tkinter (usually included with Python)

Installation

Clone or download this repository.
Install the required packages:

pip install tensorflow pillow numpy

Usage

Run the application:

python digit_recognition.py


The GUI will open with a white canvas.
Draw a digit (0-9) using your mouse (hold left-click to draw).
Click "Predict Digit" to see the model's prediction and confidence score.
Click "Clear Canvas" to reset and draw another digit.

Project Structure

digit_recognition.py: Main application script containing the GUI and model logic.
mnist_model.h5: Pre-trained model file (generated after first run).

How It Works

The model is a CNN with two convolutional layers, max-pooling, and dense layers.
If mnist_model.h5 exists, it loads the pre-trained model; otherwise, it trains a new model on the MNIST dataset and saves it.
Drawings are resized to 28x28 pixels (MNIST format) before prediction.
Predictions include the recognized digit and confidence score.
Error handling ensures the application doesn't crash on model or prediction failures.

Notes

The model is trained for 5 epochs by default. Adjust the epochs parameter in load_or_train_model() for better accuracy if needed.
Ensure sufficient disk space for saving the model file (~1MB).
The GUI includes smooth drawing with rounded lines for better user experience.
If errors occur, check the console for details or ensure dependencies are correctly installed.

License
MIT License
