import cv2 as cv

# Read image from your local file system
original_image = cv.imread('.\Programming\Python\RealPythonTutorials\Face-detection\pexels-photo-1648387.jpeg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('C:\Users\ljack099\.virtualenvs\Face-detection-8ODCR4ak\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
detected_faces = face_cascade.detectMultiScale(grayscale_image)

