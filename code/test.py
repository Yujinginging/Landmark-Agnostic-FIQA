import urllib.request
import os
import dlib
import bz2
import shutil
import math
import cv2
import glob
import numpy as np

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


#load from the folder
def get_images_from_folder(folder_path, extensions=['jpg', 'jpeg', 'png']):
   
    images = []
    for extension in extensions:
        pattern = os.path.join(folder_path, f'*.{extension}')
        images.extend(glob.glob(pattern))
    return images

def sigmoid(x,a = 0 , b = 1):
    return 1 / (1 + np.exp(-(x - a) / b))

def get_dlib_output(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Face detection
    faces = detector(image)

# Facial landmark detection and distance calculation
    for face in faces:
    # Get facial landmarks
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # compute Head size D
        L60 = landmarks[36] # left eye outer corner
        L64 = landmarks[39] # left eye inner corner
        L68 = landmarks[42] # right eye inner corner
        L72 = landmarks[45] # right eye outer corner 
        L16 = landmarks[8] # chin 
        midpoint_chin= (((L60 + L64 + L68 + L72)))
        print(L60)
        print(L64)
        print(L68)
        print(L72)
        print(f"midpoint_chin: {midpoint_chin} pixels")


# Load and preprocess image
# image_path = '/home/jing/FIQA_repo/img_test/518_2.jpg'
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/dlib_test'
image_files = get_images_from_folder(folder_path)

for image_file in image_files:
    get_dlib_output(image_file)