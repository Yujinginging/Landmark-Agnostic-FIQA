import urllib.request
import os
import dlib
import bz2
import shutil
import math
import cv2
import glob

def download_and_extract_shape_predictor():
    shape_predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    shape_predictor_archive_path = "shape_predictor_68_face_landmarks.dat.bz2"
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(shape_predictor_path):
        # Download shape predictor model
        print("Downloading shape predictor model...")
        urllib.request.urlretrieve(shape_predictor_url, shape_predictor_archive_path)
        print("Download complete!")

        # Extract the compressed file
        with bz2.open(shape_predictor_archive_path, 'rb') as source, open(shape_predictor_path, 'wb') as dest:
            shutil.copyfileobj(source, dest)
        
        print("Extraction complete!")

    return shape_predictor_path

# Load face detector
detector = dlib.get_frontal_face_detector()

# Download and extract shape predictor model
predictor_path = download_and_extract_shape_predictor()

# Load the shape predictor
predictor = dlib.shape_predictor(predictor_path)

# Load an image
image_path = "/home/teakoo/Landmark-Agnostic-FIQA/img_test/518.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Loop over each detected face
for face in faces:
    # Use the shape predictor to find facial landmarks
    shape = predictor(gray, face)

    # Draw the facial landmarks on the image with point numbers
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        #save the image with rectangles: for test and view results later
    cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_3_withlandmark.jpg',image)
    
# Display the result
cv2.imshow("Facial Landmarks with Numbers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

