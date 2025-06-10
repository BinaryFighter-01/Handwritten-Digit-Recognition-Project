# Import all required libraries 
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load or train the MNIST model with error handling
def load_or_train_model():
    model_path = 'mnist_model.h5'
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("Loaded pre-trained model.")
        else:
            print("Training new model...")
            # Load MNIST dataset
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

            # Define model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5, batch_size=32)
            model.save(model_path)
            print("Model trained and saved.")
        return model
    except Exception as e:
        print(f"Error loading/training model: {e}")
        return None

# Main application class
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.model = load_or_train_model()
        if self.model is None:
            tk.messagebox.showerror("Error", "Failed to load or train model. Exiting.")
            root.destroy()
            return

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)

        # Prediction label
        self.pred_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
        self.pred_label.pack(pady=10)

        # Buttons
        self.clear_button = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        self.predict_button = tk.Button(root, text="Predict Digit", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # Initialize drawing
        self.image = Image.new("L", (200, 200), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=8, capstyle=tk.ROUND, smooth=True)
            self.draw_image.line([self.last_x, self.last_y, x, y], fill=0, width=8)
        self.last_x, self.last_y = x, y

    def reset_draw(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Prediction: None")
        self.last_x, self.last_y = None, None

    def predict(self):
        try:
            # Resize image to 28x28 for MNIST model
            img = self.image.resize((28, 28))
            img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

            # Predict digit
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            self.pred_label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2f})")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Prediction failed: {e}")

# Run the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DigitRecognizerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
