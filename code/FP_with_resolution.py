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

def eye_rectangles(parsing, eye_class_id, original_image):
    # Given a parsing map and the class ID corresponding to eyes, 
    # extract the positions of the eyes
    eye_mask = (parsing == eye_class_id).astype(np.uint8)
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the positions of the eyes and draw rectangles
    eye_positions = []
    eye_rectangles = []
    
    # Convert the original image to a NumPy array
    image_np = np.array(original_image)

    for contour in contours:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            eye_positions.append((cx, cy))
            
            # Calculate the bounding rectangle around the eye contour
            x, y, w, h = cv2.boundingRect(contour)
            eye_rectangles.append((x, y, x + w, y + h))
            
            # Draw the rectangle on the original image
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Display the image with rectangles around the eyes using OpenCV
    # cv2.imshow('Image with Eye Rectangles', image_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Return the two rectangles as r1 and r2
    #print(eye_rectangles)
    if len(eye_rectangles) >= 2:
        r1, r2 = eye_rectangles[:2]
        return r1, r2
    else:
        r1 = eye_rectangles[0]
        return r1
    

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
    # Given a parsing map and the class ID corresponding to the chin,
    # extract the position of the chin (lowest point)
    chin_mask = (parsing == chin_class_id).astype(np.uint8)
    contours, _ = cv2.findContours(chin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the position of the chin (lowest point)
    chin_position = None
    max_y = 0  # Initialize with a low value
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if y > max_y:
                max_y = y
                chin_position = (x, y)

    return chin_position

def visualize_eye_positions(image, eye_positions):
    # Visualize the eye positions on the image
    vis_image = image.copy()
    for (cx, cy) in eye_positions:
        cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), 2)  # Green circle for eye positions

    # Display the result
    cv2.imshow("Eye Positions", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate(image_path, cp):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.open(image_path)
    image = img.resize((1350, 1350), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    out = net(img)[0]
    parsing = out.detach().squeeze(0).cpu().numpy().argmax(0)

    # Extract positions of left and right eyes 
    left_eye_positions,counters = extract_eye_positions(parsing, eye_class_id=4)
    right_eye_positions,counters = extract_eye_positions(parsing, eye_class_id=5)
    if len(left_eye_positions)>=2:
        left_eye_position = left_eye_positions[0]
        right_eye_position = left_eye_positions[1]
        R1,R2=eye_rectangles(parsing, eye_class_id=4,original_image=image)
         # Calculate the center positions
        #left_eye_center = tuple(np.round(np.mean(left_eye_positions, axis=0)).astype(int))
    elif len(left_eye_positions) == 1 and len(right_eye_positions)>=1:
        left_eye_position = left_eye_positions[0]
        
        right_eye_position = right_eye_positions[0]
        center_x = (left_eye_position[0] + right_eye_position[0]) / 2
        center_y = (left_eye_position[1] + right_eye_position[1]) / 2
        left_eye_center = (center_x,center_y)
        R1 = eye_rectangles(parsing, eye_class_id=4,original_image=image)
        R2 = eye_rectangles(parsing, eye_class_id=5,original_image=image)
    elif len(left_eye_positions) ==0 and len(right_eye_positions)>=2:
        left_eye_position = right_eye_positions[0]
        right_eye_position = right_eye_positions[1]
        R1,R2=eye_rectangles(parsing, eye_class_id=5,original_image=image)
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
   
    return R1,R2,left_eye_center,chin_point,left_eye_position,right_eye_position,counters,parsing

def sigmoid(x,a,b):
    return 1 / (1 + np.exp(-(x - a) / b))



#methods to get raw data and different QCs
#distance: head size
def get_T(eye_center, chin_point):
    L_eye = np.array(eye_center)
    L16 = np.array(chin_point)
    T = np.linalg.norm(eye_center-L16)
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

def get_IED(left_eye,right_eye):
    X_L=left_eye[0]
    Y_L=left_eye[1]
    X_R=right_eye[0]
    Y_R=right_eye[1]
    
    eyes_middle_point_X = (X_L + X_R)/2
    eyes_middle_point_Y = (Y_L +Y_R)/2
    eyes_middle_point = (eyes_middle_point_X,eyes_middle_point_Y)
    
    firstpart= (X_L - X_R) **2
    secondpart = (Y_L - Y_R) **2
    IED=np.sqrt(firstpart + secondpart) #by default set1/cos = 1
    #compute the IED quality component
    IED_QC = round(100 * (sigmoid(IED,70,20)))
    return IED, IED_QC,eyes_middle_point
    
  
def eye_open(left_eye,right_eye,counters,T):
    #upper & lower
    #get upper and lower points for both left and right eyes
    left_upper_eye,left_lower_eye=find_upper_lower(left_eye,counters)
    right_upper_eye, right_lower_eye = find_upper_lower(right_eye,counters)
        
    #DL &DR
    DL = np.abs(left_lower_eye[1] - left_upper_eye[1])
    DR = np.abs(right_upper_eye[1] - right_lower_eye[1])
    
    D_PAL = min(DL,DR)
        
    #eyes open
    
    w_eye= D_PAL / T
    #print('w (eye openness): ',w_eye)
    # compute the quality component of eye openness
    eye_openness_QC = round(100 * sigmoid(w_eye, 0.02, 0.01))
    return w_eye,eye_openness_QC


def mouths_closed(parsing,T):
    #lip upper_lower position
    upper_lip_positions,counter_lip = extract_eye_positions(parsing,eye_class_id=12)
    #print('lip upper',upper_lip_positions)
    lower_lip_positions,counter_lip = extract_eye_positions(parsing,eye_class_id=13)
    #print('lower lip:',lower_lip_positions)
    
    length = min(len(upper_lip_positions),len(lower_lip_positions))
    # Use only the first 2 points for upper and lower lips
    upper_lip_positions = upper_lip_positions[:length]
    lower_lip_positions = lower_lip_positions[:length]
    D_Lmouth= np.linalg.norm(np.abs( np.array(upper_lip_positions) - np.array(lower_lip_positions)))
    
    w_mouth = D_Lmouth / T
    
    # compute mouth closed quality component
    mouth_closed_QC = round(100 * (1 - sigmoid(w_mouth, 0.2, 0.06)))

    #print(f"DL_mouth: {D_Lmouth}")
    #print(f"W_mouth: {w_mouth}")
    #print(f" mouth_closed_QC: {mouth_closed_QC}")
    return w_mouth, mouth_closed_QC


def face_crops(right_eye,left_eye,eyes_middle_point,A,B,IED):
    # compute the leftward crop QC of the face in image
    leftward_crop_QC = round(100 * (sigmoid(right_eye[0]/IED, 0.9, 0.1)))
    #print(f"leftward_crop_QC = {leftward_crop_QC}")

    # compute the rightward crop QC of the face in image
    rightward_crop_QC = round(100 * (sigmoid((A - left_eye[0])/IED, 0.9, 0.1)))
    #print(f"rightward_crop_QC = {rightward_crop_QC}") 

    # compute the downward crop QC of the face in image
    downward_crop_QC = round(100 * (sigmoid(eyes_middle_point[1]/IED, 1.4, 0.1)))
    #print(f"dwonward_crop_QC = {downward_crop_QC}")  

    # compute the upward crop QC of the face in image
    upward_crop_QC = round(100 * (sigmoid((B - eyes_middle_point[1])/IED, 1.8, 0.1)))
    #print(f"upward_crop_QC = {upward_crop_QC}")     
    return leftward_crop_QC,rightward_crop_QC,downward_crop_QC,upward_crop_QC

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
    resized_mask = transforms.Resize((424, 424), Image.NEAREST)(segmentation_mask.unsqueeze(0)).squeeze()

    # Step 9: Pad the segmentation mask by 96 pixels from all sides
    padded_mask = nn.ZeroPad2d(96)(resized_mask.unsqueeze(0)).squeeze()

    return padded_mask

def extract_eye_zone(segmentation_mask, eye_class_id):
    if segmentation_mask is None:
        return None

    # Convert the segmentation mask to a NumPy array
    segmentation_np = segmentation_mask.cpu().numpy()

    # Create a binary mask for the eye class
    eye_mask = (segmentation_np == eye_class_id).astype(np.uint8)

    # Find contours of the eye mask
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the eyes
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        eye_zone = (x, y, x + w, y + h)
        return eye_zone
    
def expand_rectangles(r1, r2, V):
    # Ensure r1 and r2 are not None
    if  r2 is None:
        x1, y1, x2, y2 = r1
        expanded_r1 = (x1 - V, y1 - V, x2 + V, y2 + V)
        return expanded_r1, r2
    elif r1 is not None and r2 is not None:
    # Extract coordinates from rectangles
        x1, y1, x2, y2 = r1
        x3, y3, x4, y4 = r2

    # Expand rectangles on all sides by V pixels
        expanded_r1 = (x1 - V, y1 - V, x2 + V, y2 + V)
        expanded_r2 = (x3 - V, y3 - V, x4 + V, y4 + V)

        return expanded_r1, expanded_r2

def eye_visible(image_path,IED,R1,R2):
     #eye visible
    #model
    
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
    weights = torch.load('/home/jing/FIQA_repo/FaceExtraction/epoch_16_best.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model.load_state_dict(new_weights)
    #model.to(DEVICE)
    model.eval()

    #segmentation mask M
    M = face_occlusion_segmentation(image_path,model=model)
    O = extract_eye_zone(M,eye_class_id=4)
    #V - offset distance
    V = IED/20
    
    
    expanded_r1, expanded_r2 = expand_rectangles(R1, R2, V)
    if expanded_r2 is None:
        return None
    else:
        E = expanded_r1 + expanded_r2
    E = torch.tensor(E)
    #print('E:',E)
    #print('O:',O)
    if O is None:
        alpha = 1
    else:
        alpha = torch.abs(E-O)/torch.abs(E)
    #alpha_np = alpha.numpy()
    #print('alpha:',alpha)
    
    #eye visible QC
    QC_EV = round(100*alpha)
    #print('Eye visible QC: ',QC_EV)
    return QC_EV



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



         
def get_all_QCs(folder_path,excel_file_path):
    list_A=[]
    list_B= []
    list_D= []
    list_T = []
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
    list_QC_EV = []
    
    #image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))] #for all types of img files

    image_files = read_images_in_folder_and_subfolders(folder_path=folder_path)
    print(image_files)
    
    images_path = []
    for image_file in image_files:
        #image_path = os.path.join(folder_path, image_file)
        images_path.append(image_file)

        R1,R2,eye_center,chin_point,left_eye,right_eye,counters,parsing = evaluate(image_file, cp='/home/jing/FIQA_repo/face_parsing_PyTorch/res/cp/79999_iter.pth')

        if R1 is None or R2 is None or left_eye is None or right_eye is None:
            images_path.remove(image_file)
        else:
        #get T
            T = get_T(eye_center,chin_point)
        
            A,B,D,QC = get_heights_weights_QC(image_file,T)
       
        
        #IED
            IED,IED_QC,eyes_middle_point = get_IED(left_eye,right_eye)
        
        
        #eye open
            w_eye,eye_openness_QC = eye_open(left_eye,right_eye,counters,T)
        
        
        #mouth closed
            w_mouth, mouth_closed_QC = mouths_closed(parsing,T)
       
        
        #face crops
            leftward_crop_QC,rightward_crop_QC,downward_crop_QC,upward_crop_QC = face_crops(right_eye,left_eye,eyes_middle_point,A,B,IED)
        
        
        #eye visible
            #QC_EV = eye_visible(image_path,IED=IED,R1=R1,R2=R2)
        
            #add to list
            list_A.append(A)
            list_B.append(B)
            list_D.append(D)
            list_T.append(T)
        #head_size QCs:
            list_headsize_QC.append(QC)
            list_w_eye.append(w_eye)
            list_eye_openness_QC.append(eye_openness_QC)
            list_IED.append(IED)
            list_IED_QC.append(IED_QC)
            list_w_mouth.append(w_mouth)
            list_mouth_closed_QC.append(mouth_closed_QC)
            list_left_crop_QC.append(leftward_crop_QC)
            list_right_crop_QC.append(rightward_crop_QC)
            list_up_crop_QC.append(upward_crop_QC)
            list_down_crop_QC.append(downward_crop_QC)
            #list_QC_EV.append(QC_EV)
    
    #store all lists in an excel file
    data={'image paths':images_path,'A':list_A,'B':list_B,'D':list_D,'T': list_T, 'Head size QC':list_headsize_QC,'IED':list_IED,'IED QC':list_IED_QC,
          'eye openness':list_w_eye, 'eye openness QC':list_eye_openness_QC, 'mouth closed':list_w_mouth, 'mouth closed QC':list_mouth_closed_QC,
          'left crop QC':list_left_crop_QC,'right crop QC':list_right_crop_QC,'up crop QC':list_up_crop_QC,'down crop QC':list_down_crop_QC,
          }#'eye visible QC': list_QC_EV}
    df = pd.DataFrame(data)
    #excel_file_path = '/home/jing/FIQA_repo/code/excel outputs/face_parsing_outputs_lfw-deepfunneled.csv'
    
    # Save the DataFrame to an Excel file
    df.to_csv(excel_file_path)
    
    
#get_all_QCs('/home/jing/FIQA_repo/datasets/neutral_front/neutral_front')
#get_all_QCs('/home/jing/FIQA_repo/datasets/archive/lfw-deepfunneled/lfw-deepfunneled')

input_folder = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front'
output_folder_min = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front_min'
output_folder_max = '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front_max'
excel_1 = '/home/jing/FIQA_repo/code/excel outputs/face_parsing_outputs_NF_min.csv'
excel_2 = '/home/jing/FIQA_repo/code/excel outputs/face_parsing_outputs_NF_max.csv'

resize_and_save(input_folder,output_folder=output_folder_min,rate=1/2)
get_all_QCs(output_folder_min,excel_1)

resize_and_save(input_folder,output_folder_max, 6/4)
get_all_QCs(output_folder_max,excel_2)
