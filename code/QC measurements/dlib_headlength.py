import urllib.request
import pandas as pd
import os
import dlib
import bz2
import shutil
import cv2
import numpy as np
from PIL import Image


def resize_and_save(input_folder, output_folder, rate):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Construct the full path to the input image
            input_image_path = os.path.join(input_folder, filename)

            # Open the image
            image_file = Image.open(input_image_path)

            # Get the width and height of the picture
            width, height = image_file.size
            width1 = round(width * rate)
            height1 = round(height * rate)

            # Resize the image
            new_image = image_file.resize((width1, height1))

            # Construct the full path to the output image
            output_image_path = os.path.join(output_folder, filename)

            # Save the resized image
            new_image.save(output_image_path)


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



# methods to get the raw value of the QC and their mapped values

# coordinates of the left eye center
def get_left_eye_center(L60 , L64):
    left_eye_center_point = ((L60[0] + L64[0]) // 2, (L60[1] + L64[1]) // 2)
    return left_eye_center_point

# coordinates of the right eye center
def get_right_eye_center(L68 , L72):
    right_eye_center_point = ((L68[0] + L72[0]) // 2, (L68[1] + L72[1]) // 2)
    return right_eye_center_point

# Coordinates of the midpointes between the eyes
def get_eye_center(left_eye_center_point,right_eye_center_point):
    center_point = ((left_eye_center_point[0] + right_eye_center_point[0]) // 2,
                     (left_eye_center_point[1] + right_eye_center_point[1]) // 2)
    return center_point


# Distance between the Chin and midpoint between the eyes
def get_T(eye_center, chin_point):
    L_eye = np.array(eye_center)
    L16 = np.array(chin_point)
    T = np.linalg.norm(eye_center-L16)
    return T

def get_dlib_output(image_path, excel_path):
    list_A=[]
    list_B= []
    list_head_length= []
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))] #for all types of img files


    images_path = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        images_path.append(image_path)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        # Face detection
        faces = detector(image)
        # Facial landmark detection and distance calculation
        if not faces:
            images_path.remove(image_path)
        else:
             for face in faces:

    # Get facial landmarks
               shape = predictor(gray, face)
               landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # get landmarks to compute components
               L60 = landmarks[45] # right eye outer corner
               L64 = landmarks[42] # right eye inner corner
               L68 = landmarks[39] # left eye inner corner
               L72 = landmarks[36] # left eye outer corner 
               L16 = landmarks[8] # chin 

    # calculate left eye center
               left_eye_center_point = get_left_eye_center(L60 , L64)

    # claculate the right eye center 
               right_eye_center_point = get_right_eye_center(L68 , L72)

    # Calculate the center point between left and right eyes
               center_point = get_eye_center(left_eye_center_point,right_eye_center_point)   

    # the image height and width
               B = image.shape[0] # the image height
               A = image.shape[1] # the image width

    # comput the Head size quality component
               T = get_T(center_point, L16) # the distance_midpoint between eyes to chin
               head_length = 2 * T # as the painting method proposed

    #add to list
               list_A.append(A)
               list_B.append(B)
               list_head_length.append(head_length)

    #store all lists in an excel file
    data={'image paths':images_path,'A':list_A,'B':list_B,'Head Length':list_head_length}
    df = pd.DataFrame(data)
    excel_file_path = excel_path
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path)

# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/neutral_front'
excel_path = '/home/teakoo/Landmark-Agnostic-FIQA/code/excel_output/dlib_neutral_front_Head_length_outputs.csv'

get_dlib_output(folder_path, excel_path)

# minimize resolutuion to half
input_folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/neutral_front'
output_folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/neutral_front_min'

resize_and_save(input_folder_path, output_folder_path, 1/2)

# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/neutral_front_min'
excel_path = '/home/teakoo/Landmark-Agnostic-FIQA/code/excel_output/dlib_neutral_min_front_Head_length_outputs.csv'

get_dlib_output(folder_path, excel_path)

# maximize resolutuion to 150% of original size
input_folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/neutral_front'
output_folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/neutral_front_max'

resize_and_save(input_folder_path, output_folder_path, 6/4)

# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/neutral_front_max'
excel_path = '/home/teakoo/Landmark-Agnostic-FIQA/code/excel_output/dlib_neutral_max_front_Head_length_outputs.csv'

get_dlib_output(folder_path, excel_path)