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

def sigmoid(x,x0,w0):
    return (1 + np.exp((x0 - x) / w0)) ** (-1)

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

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
        left_eye_center_point = ((L60[0] + L64[0]) // 2, (L60[1] + L64[1]) // 2)

    # claculate the right eye center 
        right_eye_center_point = ((L68[0] + L72[0]) // 2, (L68[1] + L72[1]) // 2)

    # Calculate the center point between left and right eyes
        center_point = ((left_eye_center_point[0] + right_eye_center_point[0]) // 2, (left_eye_center_point[1] + right_eye_center_point[1]) // 2)
        print(center_point)    
    #distance_center eye to chin
        distance_centereye_chin =  math.sqrt((center_point[0] - L16[0]) ** 2 + (center_point[1] - L16[1]) ** 2)
    # the image height and width
        B = image.shape[0] # the image height
        A = image.shape[1] # the image width

    # comput the Head size
        T = np.linalg.norm(np.array(center_point) - np.array(L16)) # the distance_center eye to chin
        D = T / B # the head size 

    # compute head size quality component
        sigmoid_value_0 = sigmoid(np.abs(D - 75),0,5)
        print(f"sigmoid 0:{sigmoid(75,0,5)}")
        print(f"sigmoid 1:{sigmoid(74,0,5)}")

        sigmoid_value_0_0= truncate(sigmoid_value_0,0)
        print(f"sigmoid value:{sigmoid_value_0}")
        print(f"sigmoid value truncated:{sigmoid_value_0_0}")
        sigmoid_value_1 = 1 - sigmoid_value_0_0

        print(f"1 minus sigmoid value:{sigmoid_value_1}")
        sigmoid_value_2 = 100 * (sigmoid_value_1 )
        print(f"1 minus sigmoid value multiply by 100:{sigmoid_value_2}")
        final = round(sigmoid_value_2 )
        print(f"final:{final}")

        head_size_QC = round(100 * (1 - (sigmoid_value_0 )))
        print(f"T: {T} pixels")
        print(f"D: {D} pixels")
        print(f"B: {B} pixels")
        print(f" head_size_QC: { head_size_QC}")

    # compute the IED inter eye distance 
        IED =  math.sqrt((left_eye_center_point[0] - right_eye_center_point[0]) ** 2 + (left_eye_center_point[1] - right_eye_center_point[1]) ** 2)
        print(f"IED: {IED} pixels")

    # compute the IED quality component
        IED_QC = round(100 * (sigmoid(IED,70,20)))
        print(f"IED_QC: {IED_QC}")

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

        print(f"DL: {DL}")
        print(f"DR: {DR}")
        print(f"DPAL: {DPAL}")
        print(f"W: {W_eye}")
        print(f"eye opennes QC: {eye_openness_QC}")

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

        print(f"DL_mouth: {DL_mouth}")
        print(f"W_mouth: {W_mouth}")
        print(f" mouth_closed_QC: {mouth_closed_QC}")


    # compute the leftward crop QC of the face in image
        leftward_crop_QC = round(100 * (sigmoid(right_eye_center_point[0]/IED, 0.9, 0.1)))
        print(f"leftward_crop_QC = {leftward_crop_QC}")

    # compute the rightward crop QC of the face in image
        rightward_crop_QC = round(100 * (sigmoid((A - left_eye_center_point[0])/IED, 0.9, 0.1)))
        print(f"rightward_crop_QC = {rightward_crop_QC}") 

    # compute the downward crop QC of the face in image
        downward_crop_QC = round(100 * (sigmoid(center_point[1]/IED, 1.4, 0.1)))
        print(f"dwonward_crop_QC = {downward_crop_QC}")  

    # compute the upward crop QC of the face in image
        upward_crop_QC = round(100 * (sigmoid((B - center_point[1])/IED, 1.8, 0.1)))
        print(f"upward_crop_QC = {upward_crop_QC}")     

    #draw the lines on the picture
        cv2.line(image,center_point,L16,(0,255,0),2)
        cv2.putText(image, "T" , (center_point[0], center_point[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.line(image, left_eye_center_point, right_eye_center_point, (0,255,0), 2)
        cv2.putText(image, "IED" , (left_eye_center_point[0], left_eye_center_point[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.line(image,center_point,L16,(0,255,0),2)
        if DPAL == left_upper_lower_eyelid_distance_1:
            cv2.line(image,left_upper_eyelid_1,left_lower_eyelid_1,(0,255,0),2)
            cv2.putText(image, "DPAL" , (left_upper_eyelid_1[0], left_upper_eyelid_1[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DPAL == left_upper_lower_eyelid_distance_2:
            cv2.line(image,left_upper_eyelid_2,left_lower_eyelid_2,(0,255,0),2)
            cv2.putText(image, "DPAL" , (left_upper_eyelid_2[0], left_upper_eyelid_2[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DPAL == right_upper_lower_eyelid_distance_1:
            cv2.line(image,right_upper_eyelid_1,right_lower_eyelid_1,(0,255,0),2)
            cv2.putText(image, "DPAL" , (right_upper_eyelid_1[0], right_upper_eyelid_1[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DPAL == right_upper_lower_eyelid_distance_2:
            cv2.line(image,right_upper_eyelid_2,right_lower_eyelid_2,(0,255,0),2)
            cv2.putText(image, "DPAL" , (right_upper_eyelid_2[0], right_upper_eyelid_2[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DL_mouth == DL1:
            cv2.line(image,L89,L95,(0,255,0),2)
            cv2.putText(image, "DL_mouth" , (L89[0], L89[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DL_mouth == DL2:
            cv2.line(image,L90,L94,(0,255,0),2)
            cv2.putText(image, "DL_mouth" , (L90[0], L90[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if DL_mouth == DL3:
            cv2.line(image,L91,L93,(0,255,0),2)
            cv2.putText(image, "DL_mouth" , (L91[0], L91[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    


    # Save the image with rectangles and landmarks
        output_path = image_path.replace('.jpg', '_dlib_output.jpg')
        cv2.imwrite(output_path, image)
        print(f"Image saved with rectangles and landmarks: {output_path}")


# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/dlib_test'
image_files = get_images_from_folder(folder_path)

for image_file in image_files:
    get_dlib_output(image_file)