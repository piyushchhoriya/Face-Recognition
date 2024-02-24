import os
import cv2
import numpy as np

# Function to get the images and label data
def get_images_and_labels(data_path):
    # Initialize lists to store face images and corresponding labels
    face_images = []
    labels = []

    # Iterate through the folders in the data path
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # Load the image using OpenCV
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Get the label from the folder name
            label = int(os.path.basename(root))
            
            # Append the image and label to the lists
            face_images.append(img)
            labels.append(label)

    return face_images, labels

# Path to the directory containing training data
training_data_path = 'faces'

# Get the face images and labels
images, labels = get_images_and_labels(training_data_path)

# Initialize the LBPH face recognizer
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Train the recognizer with the face images and labels
recognizer.train(images, np.array(labels))

# Save the trained model to a file
recognizer.save('trained_model.yml')

print("Training completed. Model saved as 'trained_model.yml'")
