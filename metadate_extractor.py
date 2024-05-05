import os
import cv2
import numpy as np
import pandas as pd

list = []
def extract_features(image, label):
    # Load an image
    img_path = image
    img = cv2.imread(img_path)

    # Get image dimensions (height, width) and number of channels
    height, width, channels = img.shape
    metadata_dict = {
        "Image_Path": img_path,
        "Height": height,
        "Width": width,
        "Channels": channels,
        "Class": label
    }
    list.append(metadata_dict)

for file in os.listdir("data/vehicles"):
    extract_features("data/vehicles/"+file, 1)

for file in os.listdir("data/non-vehicles"):
    extract_features("data/non-vehicles/"+file, 0)

df = pd.DataFrame.from_dict(list)

df.to_csv("annotations.csv", index=False)