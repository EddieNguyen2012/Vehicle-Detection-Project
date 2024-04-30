import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('data/annotations.csv')

def preprocess_image(image_path, target_size=(224, 224)):
    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Resize image
    image = cv2.resize(image, target_size)

    # Convert to numpy array and normalize pixel values
    image_array = image.astype(np.float32) / 255.0

    return image_array


# Example usage:
image_path = 'path/to/your/image.jpg'
preprocessed_image = preprocess_image(image_path)