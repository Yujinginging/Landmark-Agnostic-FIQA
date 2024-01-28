from logger import setup_logger
from model import BiSeNet

import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
import segmentation_models_pytorch as smp
from collections import OrderedDict
import torch.nn as nn

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
            
#the previous methods
def find_upper_lower(eye_position, contours):
    upper_point = None
    lower_point = None

    # Get x-coordinate from the eye position
    eye_x = eye_position[0]
    
    # Initialize y-coordinates with extreme values
    up_y = 0
    low_y = 400

    # Iterate through all contours to find upper and lower points along the vertical line
    for contour in contours:
        for point in contour[:, 0]:  # Accessing the points correctly
            x, y = point

            up_y = max(up_y, y)
            low_y = min(low_y, y)

    upper_point = (eye_x, up_y)
    lower_point = (eye_x, low_y)

    return upper_point, lower_point


    

def extract_eye_positions(parsing, eye_class_id):
    # Given a parsing map and the class ID corresponding to eyes, 
    # extract the positions of the eyes
    eye_mask = (parsing == eye_class_id).astype(np.uint8)
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the positions of the eyes
    eye_positions = []
    #print(contours)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            eye_positions.append((cx, cy))

    return eye_positions,contours


def extract_chin_position(parsing, chin_class_id):
   # Create a binary mask for the chin class
    chin_mask = (parsing == chin_class_id).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(chin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assumed to be the chin)
    max_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the chin contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Calculate the chin point as the bottom-center of the bounding box
    chin_point = (x + w // 2, y + h)

    return chin_point


def visualize_eye_positions(image, eye_positions):
    # Visualize the eye positions on the image
    vis_image = image.copy()
    for (cx, cy) in eye_positions:
        cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), 2)  # Green circle for eye positions

    # Display the result
    cv2.imshow("Eye Positions", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate(image_path, cp,num):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    # to_tensor = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    to_tensor = transforms.ToTensor()

    img = Image.open(image_path)
    image = img.resize((num,num), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    #net = net.to('cuda') #move model to the GPU otherwise the previous results were all none T_T
    out = net(img)[0]
    parsing = out.detach().squeeze(0).cpu().numpy().argmax(0)

    # Extract positions of left and right eyes 
    left_eye_positions,counters = extract_eye_positions(parsing, eye_class_id=4)
    right_eye_positions,counters = extract_eye_positions(parsing, eye_class_id=5)
    if len(left_eye_positions)>=2:
        left_eye_position = left_eye_positions[0]
        right_eye_position = left_eye_positions[1]
         # Calculate the center positions
        #left_eye_center = tuple(np.round(np.mean(left_eye_positions, axis=0)).astype(int))
    elif len(left_eye_positions) == 1 and len(right_eye_positions)>=1:
        left_eye_position = left_eye_positions[0]
        
        right_eye_position = right_eye_positions[0]
        center_x = (left_eye_position[0] + right_eye_position[0]) / 2
        center_y = (left_eye_position[1] + right_eye_position[1]) / 2
        left_eye_center = (center_x,center_y)
    elif len(left_eye_positions) ==0 and len(right_eye_positions)>=2:
        left_eye_position = right_eye_positions[0]
        right_eye_position = right_eye_positions[1]
    else:
        left_eye_position = None
        right_eye_position = None
        R1=None
        R2=None
    # a pair of positions[left,right]
    
    #chin
    chin_point = extract_chin_position(parsing,chin_class_id=1)
    #print ('chin:',chin_point)
    
    # Visualize eye positions on the image
    
    #visualize_eye_positions(np.array(image), left_eye_positions)
    if left_eye_position is not None:
        center_x = (left_eye_position[0] + right_eye_position[0]) / 2
        center_y = (left_eye_position[1] + right_eye_position[1]) / 2
        left_eye_center = (center_x,center_y)
    else:
        left_eye_center = None
    #print('Eye Center:', left_eye_center)
   
    return left_eye_center,chin_point,left_eye_position,right_eye_position,counters,parsing

def sigmoid(x,a,b):
    return 1 / (1 + np.exp(-(x - a) / b))



#methods to get raw data and different QCs
#distance: head size
def get_T(eye_center, chin_point):
    L_eye = np.array(eye_center)
    L16 = np.array(chin_point)
    T = np.linalg.norm(L_eye-L16)
    return T
    #height & width


    
    
#methods to get all images
def read_images_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            with Image.open(image_path) as img:
                images.append(img)
                # Do something with the image if needed
        except Exception as e:
            print(f"Error reading {image_path}: {e}")

    return images
    
    
#get images height and weight:
def get_heights_weights_QC(image_path,T):
    
    with Image.open(image_path) as img:
        height = img.size[1]
        width = img.size[0]
    B = height
    A = width
    #print('B:', B)
    #print('A:',A)
    D=T/B
    #print('D: ' , D)
    QC=np.round(200*(1-sigmoid(np.abs(D-0.45),0,0.05))) # change is done by the file Late-FR-comment-on-CD3-29794-5-231.. from christoph
    return A, B, D, QC


#in case there are sub folders in your selected datsets
def read_images_in_folder_and_subfolders(folder_path):
    image_files = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Append the full path to the list of image files
                image_files.append(os.path.join(root, file))


    return image_files



         
def get_Headlengths(folder_path,excel_file_path,num):
    list_T=[]
   
    
    #image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))] #for all types of img files

    image_files = read_images_in_folder_and_subfolders(folder_path=folder_path)
    #print(image_files)
    
    images_path = []
    for image_file in image_files:
        #image_path = os.path.join(folder_path, image_file)
        images_path.append(image_file)

        eye_center,chin_point,left_eye,right_eye,counters,parsing = evaluate(image_file, cp='/home/jing/FIQA_repo/face_parsing_PyTorch/res/cp/79999_iter.pth',num=num)
        print(eye_center, chin_point)
        if left_eye is None or right_eye is None:
            images_path.remove(image_file)
        else:
        #get T
            T = get_T(eye_center,chin_point)
            print(T)
           
            list_T.append(T)
       
    
    #store all lists in an excel file
    data={'image paths':images_path,'T': list_T}
    df = pd.DataFrame(data)
    #excel_file_path = '/home/jing/FIQA_repo/code/excel outputs/face_parsing_outputs_lfw-deepfunneled.csv'
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path)
    
    
#get_all_QCs('/home/jing/FIQA_repo/datasets/neutral_front/neutral_front')


input_folder = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front'
output_folder_min = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front_min'
output_folder_max = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front_max'
excel_1 = '/home/jing/FIQA_repo/code/excel outputs/HLFP_NF_min.csv'
excel_2 = '/home/jing/FIQA_repo/code/excel outputs/HLFP_NF_max.csv'
excel_3 = '/home/jing/FIQA_repo/code/excel outputs/HLFP_NF.csv'

resize_and_save(input_folder,output_folder=output_folder_min,rate=1/2)
get_Headlengths(output_folder_min,excel_1,(1350*1/2))

resize_and_save(input_folder,output_folder_max, 6/4)
get_Headlengths(output_folder_max,excel_2,(1350*6/4))

#get_Headlengths(input_folder,excel_3,1350)

# get_all_QCs(input_folder,excel_3,1350)
# excel_lfw= '/home/jing/FIQA_repo/code/excel outputs/face_parsing_outputs_LFW.csv'
# get_all_QCs('/home/jing/FIQA_repo/datasets/archive/lfw-deepfunneled/lfw-deepfunneled',excel_lfw,250)