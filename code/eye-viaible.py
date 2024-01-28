import urllib.request
import pandas as pd
import os
import dlib
import bz2
import shutil
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from collections import OrderedDict
import torch.nn as nn
import math


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

def rectangle_area(vertices):
    area = 0.5 * abs(np.dot(vertices[:,0], np.roll(vertices[:,1],1)) - np.dot(vertices[:,1], np.roll(vertices[:,0],1)) )
    return area

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

def face_occlusion_segmentation(image_path,model):
    # Step 1: Crop the image from all sides by 96 pixels
    original_image = Image.open(image_path)
    cropped_image = original_image.crop((96, 96, original_image.width - 96, original_image.height - 96))

    # Step 2: Scale the image to 224x224 pixels
    resized_image = cropped_image.resize((224, 224), Image.BILINEAR)

    # Step 3: Encode the image in a 4-dimensional tensor
    tensor_image = transforms.ToTensor()(resized_image).unsqueeze(0)

    # Step 4: Divide the tensor by 255
    tensor_image /= 255.0

    # Step 5: Feed the tensor through the face parsing CNN
    with torch.no_grad():
        output_tensor = model(tensor_image)

    # Step 6: Remove the first dimension of the output tensor
    segmentation_mask = torch.argmax(output_tensor, dim=1).squeeze()

    # Step 7: Set all positive values of the segmentation mask to 1, and all other values to 0
    segmentation_mask = (segmentation_mask > 0).type(torch.float32)

    # Step 8: Resize the segmentation mask to 424x424 pixels
 #   resized_mask = transforms.Resize((424, 424), Image.NEAREST)(segmentation_mask.unsqueeze(0)).squeeze()

    # Step 9: Pad the segmentation mask by 96 pixels from all sides
 #   padded_mask = nn.ZeroPad2d(96)(resized_mask.unsqueeze(0)).squeeze()
    print(segmentation_mask.size())
    M =int

    if segmentation_mask is None:
        return None
    
    # Check if there are non-zero elements in the tensor
    if segmentation_mask.sum() > 0:
    # Find the coordinates where the value is 1
     nonzero_coords = torch.nonzero(segmentation_mask == 1)

    # Calculate the bounding box
     min_x = torch.min(nonzero_coords[:, 1])
     max_x = torch.max(nonzero_coords[:, 1])
     min_y = torch.min(nonzero_coords[:, 0])
     max_y = torch.max(nonzero_coords[:, 0])
     print(min_x)
     print(min_x.item())
     return min_x, min_y, max_x, max_y
    else:
        return None
     


    # Convert the segmentation mask to a NumPy array
    #segmentation_np = segmentation_mask.cpu().numpy()
   
    #print(segmentation_np.shape[-1])

    # Find contours of the eye mask
    #contours, _ = cv2.findContours(segmentation_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the eyes
    #if len(contours) > 0:
        #x, y, w, h = cv2.boundingRect(contours[0])
        #mask= (x, y, w, h)
        #return x, y, w, h ,mask
    
  
def find_intersection(x1,y1,width1,height1,x2,y2,width2,height2):
# Calculate coordinates of the intersection rectangle
 x_intersection = max(x1, x2)
 y_intersection = max(y1, y2)
 width_intersection = min(x1 + width1, x2 + width2) - x_intersection
 height_intersection = min(y1 + height1, y2 + height2) - y_intersection

# Ensure non-negative width and height
 width_intersection = max(0, width_intersection)
 height_intersection = max(0, height_intersection)

# Resulting coordinates of the intersection rectangle
 return x_intersection, y_intersection, width_intersection, height_intersection


  

def comput_evz(left_eye_center_point, right_eye_center_point,left_lower_eyelid_1,left_lower_eyelid_2,left_upper_eyelid_1,left_upper_eyelid_2,
               right_lower_eyelid_1, right_lower_eyelid_2,right_upper_eyelid_1,right_upper_eyelid_2, L60,L64,L68,L72,image_path):
        IED =  math.sqrt((left_eye_center_point[0] - right_eye_center_point[0]) ** 2 + (left_eye_center_point[1] - right_eye_center_point[1]) ** 2)
        V = round(IED * 5 / 100) # we assume V to be 5% of IED 
        left_eye  = [left_lower_eyelid_1, left_lower_eyelid_2, left_upper_eyelid_1, left_upper_eyelid_2, L60, L64]
        right_eye = [right_lower_eyelid_1,right_lower_eyelid_2, right_upper_eyelid_1, right_upper_eyelid_2, L68, L72]
        left_eye_coordinates = np.array(left_eye)
        right_eye_coordinates = np.array(right_eye)

        # reshape the array into a 2D array
        #left_eye_coordinates = left_eye_coordinates.reshape((-1,2))
        #right_eye_coordinates = right_eye_coordinates.reshape((-1,2))

        # find the bounding rectangle R1 and R2
        x1,y1,w1,h1 = cv2.boundingRect(left_eye_coordinates)
        x2,y2,w2,h2 = cv2.boundingRect(right_eye_coordinates)

        # maximize the R1, R2 wides and height by V
        # expand the x-coordinates by V   
        left_eye_coordinates[4,0] -= V
        left_eye_coordinates[5,0] += V
        right_eye_coordinates[4,0] -= V
        right_eye_coordinates[5,0] += V

        # expand the y-coordinates by V 
        left_eye_coordinates[0,1] += V
        left_eye_coordinates[1,1] += V
        left_eye_coordinates[2,1] -= V
        left_eye_coordinates[3,1] -= V
        right_eye_coordinates[0,1] += V
        right_eye_coordinates[1,1] += V
        right_eye_coordinates[2,1] -= V
        right_eye_coordinates[3,1] -= V

        # find the bounding rectangle R1 and R2 after maximizing with V
        x1,y1,w1,h1 = cv2.boundingRect(left_eye_coordinates)
        x2,y2,w2,h2 = cv2.boundingRect(right_eye_coordinates)

        # calculate the area E1 and E2
        E1 = rectangle_area(left_eye_coordinates)
        E2 = rectangle_area(right_eye_coordinates)

        # clculate E as E = E1 union E2
        E = E1 + E2
            
        ENCODER = 'resnet18'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = 1
        ATTENTION = None
        ACTIVATION = None
        DEVICE = 'cuda:1'
        model = smp.Unet(encoder_name=ENCODER,
                 encoder_weights=ENCODER_WEIGHTS,
                 classes=CLASSES,
                 activation=ACTIVATION)
        weights = torch.load('/home/teakoo/Landmark-Agnostic-FIQA/pretrained_model/epoch_16_best.ckpt')
        new_weights = OrderedDict()
        for key in weights.keys():
            new_key = '.'.join(key.split('.')[1:])
            new_weights[new_key] = weights[key]

        model.load_state_dict(new_weights)
        model.eval()

        #segmentation mask M
        M = face_occlusion_segmentation(image_path,model=model)
        if M is None:
         alpha = 1
         alpha_QC = round(100 * alpha)
         return alpha,  alpha_QC         
        else:
         M1,M2,M3,M4 =  M
         o1,o2,o3,o4 = find_intersection(M1,M2,M3,M4,x1,y1,w1,h1)
         o5,o6,o7,o8 = find_intersection(M1,M2,M3,M4,x2,y2,w2,h2)
         O1 = o3 * o4
         O2 = o7 * o8
         O = O1 + O2
         EO = min(E,O)
         alpha = EO / E
         alpha_QC = round(100 * alpha)
         return alpha,  alpha_QC
        

def get_dlib_output(image_path, excel_path):
    list_A=[]
    list_B= []
    list_alpha= []
    list_alpha_QC= []
    list_head_length= []
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))] #for all types of img files


    images_path = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        images_path.append(image_path)
        img = cv2.imread(image_path)
        image = cv2.resize(img,(224, 224))
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

               # comput eye visible
               alpha , alpha_QC = comput_evz(left_eye_center_point, right_eye_center_point,left_lower_eyelid_1,left_lower_eyelid_2,left_upper_eyelid_1,left_upper_eyelid_2,
               right_lower_eyelid_1, right_lower_eyelid_2,right_upper_eyelid_1,right_upper_eyelid_2, L60,L64,L68,L72,image_path)

    #add to list
               list_A.append(A)
               list_B.append(B)
               list_head_length.append(head_length)
               list_alpha.append(alpha)
               list_alpha_QC.append(alpha_QC)

    #store all lists in an excel file
    data={'image paths':images_path,'A':list_A,'B':list_B,'Head Length':list_head_length,'alpha':list_alpha,'alpha QC':list_alpha_QC}
    df = pd.DataFrame(data)
    excel_file_path = excel_path
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path)

# Load and preprocess image
folder_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/neutral_front'
excel_path = '/home/teakoo/Landmark-Agnostic-FIQA/code/excel_output/dlib_neutral_front_Head_length_outputs_1.csv'

get_dlib_output(folder_path, excel_path)