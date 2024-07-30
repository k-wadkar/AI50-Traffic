import cv2
import numpy as np
import tensorflow as tf
import sys

# If path to model and image not passed in terminal, raise error
if len(sys.argv) != 3:
    raise ValueError(
        "Pass path to model as argument 1, and path to image to classify as argument 2")

# Load the requested model
model = tf.keras.models.load_model(sys.argv[1])


def preprocess_image(image_path):
    """
    Preprocess the image to the required format.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (30, 30))
    img = np.array(img)
    img = img / 255.0  # Normalize the image if required
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def classify_image(image_path):
    """
    Classify the image using the loaded model.
    """
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return predicted_class


# Classify the image and output prediction
predicted_class = classify_image(sys.argv[2])
print(f"The predicted class is: {predicted_class}")
