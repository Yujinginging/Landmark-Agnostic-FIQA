import urllib.request
import pandas as pd
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

def sigmoid(x,x0,w0):
    return (1 + np.exp((x0 - x) / w0)) ** (-1)


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

# #get images height and weight:
# def get_height_width(image_path):
    
#     with cv2.imread(image_path) as img:
#         height = img.shape[0] # the image height
#         width = img.size[1] # the image width
#     B = height
#     A = width
#     return A, B 

# Distance between the Chin and midpoint between the eyes
def get_T(eye_center, chin_point):
    L_eye = np.array(eye_center)
    L16 = np.array(chin_point)
    T = np.linalg.norm(eye_center-L16)
    return T

# The Head Size Quality Component and mapping
def get_D_QC(T,B):
    D = D = T / B # the head size 
    D_QC=np.round(200*(1-sigmoid(np.abs(D-0.45),0,0.05))) # change is done
    return D, D_QC

# The IED Quality Component and mapping
def get_IED_QC(left_eye_center_point, right_eye_center_point):
    IED =  math.sqrt((left_eye_center_point[0] - right_eye_center_point[0]) ** 2 + (left_eye_center_point[1] - right_eye_center_point[1]) ** 2)
    IED_QC = IED_QC = round(100 * (sigmoid(IED,70,20)))
    return IED, IED_QC

# The eye open quality Component and mapping
def get_w_eye_QC(T, left_lower_eyelid_1,left_lower_eyelid_2,left_upper_eyelid_1,left_upper_eyelid_2,
                 right_lower_eyelid_1,right_lower_eyelid_2,right_upper_eyelid_1, right_upper_eyelid_2):
        # compute the upper and lower eyelids in the left eye 1 and 2
        left_upper_lower_eyelid_distance_1=  np.linalg.norm(np.array(left_upper_eyelid_1) - np.array(left_lower_eyelid_1))
        left_upper_lower_eyelid_distance_2=  np.linalg.norm(np.array(left_upper_eyelid_2) - np.array(left_lower_eyelid_2))
    
    # the largest distance between the upper and lower eyelid of left eye DL
        DL = max(left_upper_lower_eyelid_distance_1, left_upper_lower_eyelid_distance_2)

    # compute the upper and lower eyelids in the rigth eye 1 and 2
        right_upper_lower_eyelid_distance_1=  np.linalg.norm(np.array(right_upper_eyelid_1) - np.array(right_lower_eyelid_1))
        right_upper_lower_eyelid_distance_2=  np.linalg.norm(np.array(right_upper_eyelid_2) - np.array(right_lower_eyelid_2))
    
    # the largest distance between the upper and lower eyelid of right eye DR
        DR = max(right_upper_lower_eyelid_distance_1, right_upper_lower_eyelid_distance_2)

    # compute the palpebral aperture as the smaller of DL and DR
        DPAL= min(DL, DR)

    # compute the eye openness component W
        W_eye = DPAL / T

    # compute the quality component of eye openness
        eye_openness_QC = round(100 * sigmoid(W_eye, 0.02, 0.01))
        return W_eye,eye_openness_QC

# the mouth closed quality component with its mapping
def get_w_mouth_QC(T, L89, L90, L91, L93, L94, L95):
    # compute the distance between the upper and lower lip DL1, DL2, DL3
        DL1 =  np.linalg.norm(np.array(L89) - np.array(L95))
        DL2 =  np.linalg.norm(np.array(L90) - np.array(L94))
        DL3 =  np.linalg.norm(np.array(L91) - np.array(L93))

    # find the largest distance between the upper and lower lip DL
        DL_mouth = max(DL1, DL2, DL3)

    #compute the mouth closed component W_mouth
        W_mouth = DL_mouth / T

    # compute mouth closed quality component
        mouth_closed_QC = round(100 * (1 - sigmoid(W_mouth, 0.2, 0.06)))
        return W_mouth, mouth_closed_QC

# the face crop of image quality component and the mapping:
def get_face_crop_QC(A, B, IED, right_eye_center_point, left_eye_center_point, center_point):
        # compute the leftward crop QC of the face in image
        leftward_crop_QC = round(100 * (sigmoid(right_eye_center_point[0]/IED, 0.9, 0.1)))

    # compute the rightward crop QC of the face in image
        rightward_crop_QC = round(100 * (sigmoid((A - left_eye_center_point[0])/IED, 0.9, 0.1)))

    # compute the downward crop QC of the face in image
        downward_crop_QC = round(100 * (sigmoid(center_point[1]/IED, 1.4, 0.1))) 

    # compute the upward crop QC of the face in image
        upward_crop_QC = round(100 * (sigmoid((B - center_point[1])/IED, 1.8, 0.1)))
        return leftward_crop_QC, rightward_crop_QC, downward_crop_QC,upward_crop_QC

def get_dlib_output(image_path):
    list_A=[]
    list_B= []
    list_D= []
    list_headsize_QC= []
    list_IED= []
    list_IED_QC = []
    list_w_eye= []
    list_eye_openness_QC = []
    list_w_mouth= []
    list_mouth_closed_QC = []
    list_left_crop_QC= []
    list_right_crop_QC= []
    list_up_crop_QC= []
    list_down_crop_QC = []
#    list_QC_EV = []
    
   
    images_path = []
    for root, dirs, files in os.walk(folder_path):
     for file in files:
      if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        image_path = os.path.join(root, file)
        images_path.append(image_path)
        image = cv2.imread(image_path)
        if image is not None:
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
               L60 = landmarks[36] # left eye outer corner
               L64 = landmarks[39] # left eye inner corner
               L68 = landmarks[42] # right eye inner corner
               L72 = landmarks[45] # right eye outer corner 
               L16 = landmarks[8] # chin 
               left_upper_eyelid_1 = landmarks[37]
               left_upper_eyelid_2 = landmarks[38]
               left_lower_eyelid_1 = landmarks[41]
               left_lower_eyelid_2 = landmarks[40]
               right_upper_eyelid_1 = landmarks[43]
               right_upper_eyelid_2 = landmarks[44]
               right_lower_eyelid_1 = landmarks[47]
               right_lower_eyelid_2 = landmarks[46]
               L89 = landmarks[61]
               L90 = landmarks[62]
               L91 = landmarks[63]
               L95 = landmarks[67]
               L94 = landmarks[66]
               L93 = landmarks[65]


    # calculate left eye center
               left_eye_center_point = get_left_eye_center(L60 , L64)

    # claculate the right eye center 
               right_eye_center_point = get_right_eye_center(L68 , L72)

    # Calculate the center point between left and right eyes
               center_point = get_eye_center(left_eye_center_point,right_eye_center_point)
#        print(center_point)    
    #distance_center eye to chin
               distance_centereye_chin =  math.sqrt((center_point[0] - L16[0]) ** 2 + (center_point[1] - L16[1]) ** 2)

    # the image height and width
               B = image.shape[0] # the image height
               A = image.shape[1] # the image width

    # comput the Head size quality component
               T = get_T(center_point, L16) # the distance_midpoint between eyes to chin
               D, head_size_QC= get_D_QC(T,B) # the head size with the mapped value
#        print(f" head_size: { D}")
#        print(f" head_size_QC: { head_size_QC}")

    # compute the IED inter eye distance with mapping
               IED , IED_QC=  get_IED_QC(left_eye_center_point,right_eye_center_point)
#        print(f"IED: {IED} pixels")
#        print(f"IED_QC: {IED_QC}")

    # compute the eye openness quality component W_eye and its mapping
               W_eye, eye_openness_QC = get_w_eye_QC(T, left_lower_eyelid_1,left_lower_eyelid_2,left_upper_eyelid_1,left_upper_eyelid_2,
                 right_lower_eyelid_1,right_lower_eyelid_2,right_upper_eyelid_1, right_upper_eyelid_2)
#        print(f"W: {W_eye}")
#        print(f"eye opennes QC: {eye_openness_QC}")

    #compute the mouth closed component W_mouth
               W_mouth, mouth_closed_QC = get_w_mouth_QC(T, L89, L90, L91, L93, L94, L95)
#        print(f"W_mouth: {W_mouth}")
#        print(f" mouth_closed_QC: {mouth_closed_QC}")


    # compute the leftward crop QC of the face in image
               leftward_crop_QC, rightward_crop_QC, downward_crop_QC, upward_crop_QC  = get_face_crop_QC(A, B, IED, right_eye_center_point, left_eye_center_point, center_point)
#        print(f"leftward_crop_QC = {leftward_crop_QC}")
#        print(f"rightward_crop_QC = {rightward_crop_QC}") 
#        print(f"dwonward_crop_QC = {downward_crop_QC}")  
#        print(f"upward_crop_QC = {upward_crop_QC}")     

    #add to list
               list_A.append(A)
               list_B.append(B)
               list_D.append(D)
               list_headsize_QC.append(head_size_QC)
               list_w_eye.append(W_eye)
               list_eye_openness_QC.append(eye_openness_QC)
               list_IED.append(IED)
               list_IED_QC.append(IED_QC)
               list_w_mouth.append(W_mouth)
               list_mouth_closed_QC.append(mouth_closed_QC)
               list_left_crop_QC.append(leftward_crop_QC)
               list_right_crop_QC.append(rightward_crop_QC)
               list_up_crop_QC.append(upward_crop_QC)
               list_down_crop_QC.append(downward_crop_QC)
        #list_QC_EV.append(QC_EV)


    #store all lists in an excel file
    data={'image paths':images_path,'A':list_A,'B':list_B,'D':list_D,'Head size QC':list_headsize_QC,'IED':list_IED,'IED QC':list_IED_QC,
                   'eye openness':list_w_eye, 'eye openness QC':list_eye_openness_QC, 'mouth closed':list_w_mouth, 'mouth closed QC':list_mouth_closed_QC,
                    'left crop QC':list_left_crop_QC,'right crop QC':list_right_crop_QC,'up crop QC':list_up_crop_QC,'down crop QC':list_down_crop_QC,
                   }
    df = pd.DataFrame(data)
    excel_file_path = '/home/teakoo/Landmark-Agnostic-FIQA/code/excel_output/dlib_lfw-deepfunneled_outputs.csv'
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path)

# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/lfw-deepfunneled'

get_dlib_output(folder_path)