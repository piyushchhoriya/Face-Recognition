# Face-Recognition



### About the Project
This project works in three phases
1. Creating Datasets
2. Training the Model
3. Face Recognition


Creating Datasets: Gathering and organizing data for training and testing face recognition algorithms.

Training the Model: Utilizing OpenCV's models, the Haar Cascade Classifier & LBPH algorithm to train a model for face recognition.

Face Recognition: Applying the trained model for recognizing faces in images or video frames. This involves using OpenCV's Frontal-Face Haar Cascade Classifier for face detection, followed by running the LBPH algorithm for recognition and displaying the matching percentage on the screen.


### Requirements

- Python 3.6+
- OpenCV
- Numpy
- Pandas

### Taking Photos
1. Run `python CapturePhotos.py`
2. Enter your ID and Name
3. Press the 'S' key repeatedly to take photos, once a box appears around your face. It is recommended to take atleast 25 pictures.
4. Press the 'q' key when you're finished taking pictures.

### Training the Model
1. Run `python Train.py`
2. After Training is complete the program will generate the file "Classifiers/TrainedLBPH.yml" file

### Recognizing
1. Run `python Recognize.py`
It will display the matching percentage 

### Applications
1. Criminal Identification
2. Airport Security
3. Attendance Tracking
4. Healthcare - For patient identification
