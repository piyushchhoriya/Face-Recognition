import os
import numpy as np
import pandas as pd
import cv2 as cv
from datetime import datetime

# Load existing data or create a new DataFrame
if os.path.exists('id-names.csv'):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
else:
    id_names = pd.DataFrame(columns=['id', 'name'])
    id_names.to_csv('id-names.csv', index=False)  # Save an empty DataFrame if CSV doesn't exist

# Create 'faces' directory if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

print('Welcome!')
print('\nPlease input your ID.')
print('If this is your first time, choose a random ID between 1-10000')

id = int(input('ID: '))
name = ''

# Check if the ID exists in the DataFrame
if id in id_names['id'].values:
    matching_names = id_names.loc[id_names['id'] == id, 'name']
    name = matching_names.iloc[0]  # Access the first element if there are multiple matches
    print(f'Welcome Back {name}!!')
else:
    name = input('Please Enter your name: ')
    os.makedirs(f'faces/{id}')
    new_data = pd.DataFrame({'id': [id], 'name': [name]})
    id_names = pd.concat([id_names, new_data], ignore_index=True)
    id_names.to_csv('id-names.csv', index=False)  # Save DataFrame to CSV without index

print("\nLet's capture!")

print("Now this is where you begin taking photos. Once you see a rectangle around your face, press the 's' key to capture a picture.", end=" ")
print("It is recommended to take at least 20-25 pictures, from different angles, in different poses, with and without specs, you get the gist.")
input("\nPress ENTER to start when you're ready, and press the 'q' key to quit when you're done!")

camera = cv.VideoCapture(0)
face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

photos_taken = 0
while True:
    ret, img = camera.read()
    if not ret:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_region = gray[y:y + h, x:x + w]
        if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
            face_img = cv.resize(face_region, (220, 220))
            img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
            cv.imwrite(f'faces/{id}/{img_name}', face_img)
            photos_taken += 1
            print(f'{photos_taken} -> Photos taken!')

    cv.imshow('Face', img)

    # Break loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
camera.release()
cv.destroyAllWindows()
