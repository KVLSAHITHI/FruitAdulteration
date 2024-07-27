import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = r"C:\Users\srivi\OneDrive\Desktop\pythonProject\pythonProject\fruit_adulteration_model_updated.h5"
 # Replace with the actual path
model = load_model(model_path)

# Function to preprocess the input image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array


# Function to predict adulteration
def predict_adulteration(img_path):
    # Preprocess the input image
    img_array = preprocess_image(img_path)

    # Use the trained model to predict adulteration
    prediction = model.predict(img_array)

    # Interpret the prediction
    if prediction > 0.5:
        return "Adulterated"
    else:
        return "Not Adulterated"

# Sample usage
if __name__ == "__main__":
    # Get input image path from user
    img_path = input("Enter the path to the fruit image: ")

    # Check if the file exists
    if not os.path.isfile(img_path):
        print("Error: The specified file does not exist.")
    else:
        # Predict adulteration
        result = predict_adulteration(img_path)
        print(f"The fruit is predicted to be: {result}")
