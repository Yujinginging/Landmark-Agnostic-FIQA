from logger import setup_logger
from model import BiSeNet

import torch
import numpy as np
from PIL import Image,ImageDraw
import torchvision.transforms as transforms
import cv2

import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import OrderedDict

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
    print(eye_rectangles)
    if len(eye_rectangles) >= 2:
        r1, r2 = eye_rectangles[:2]
        return r1, r2
    else:
        r1 = eye_rectangles[0]
        return r1
    
    

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
    
    vis_image = image.copy()

    # Visualize the eye positions on the image
    for (cx, cy) in eye_positions:
        cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), 2)  # Green circle for eye positions

    # Normalize image pixel values if necessary
    vis_image = cv2.normalize(vis_image, None, 0, 255, cv2.NORM_MINMAX)

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
    #print(img.size())
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
    #markedpositions =[]
    #markedpositions.append(left_eye_positions[1] )
    markedpositions=left_eye_positions
    #markedpositions = markedpositions + left_eye_positions
    markedpositions.append(chin_point)
    # Visualize eye positions on the image
    #visualize_eye_positions(np.array(image), markedpositions)
    if left_eye_position is not None:
        center_x = (left_eye_position[0] + right_eye_position[0]) / 2
        center_y = (left_eye_position[1] + right_eye_position[1]) / 2
        left_eye_center = (center_x,center_y)
    else:
        left_eye_center = None
    #print('Eye Center:', left_eye_center)
    
    
    

   
    
    print(left_eye_positions)
    
    return R1,R2,left_eye_center,chin_point,left_eye_position,right_eye_position,counters,parsing

def sigmoid(x,a,b):
    return 1 / (1 + np.exp(-(x - a) / b))


from torchvision import transforms

def face_occlusion_segmentation(image_path, model):
    # Step 1: Crop the image from all sides by 96 pixels
    original_image = Image.open(image_path)
    cropped_image = original_image.crop((96, 96, original_image.width - 96, original_image.height - 96))

    # Step 2: Scale the image to 224x224 pixels
    resized_image = cropped_image.resize((224, 224), Image.BILINEAR)

    # Step 3: Encode the image in a 4-dimensional tensor
    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(resized_image).unsqueeze(0)
    print(tensor_image)
    print('dimension', tensor_image.size())

    # Step 4: Normalize the tensor (ensure it matches the normalization used during training)
    tensor_image /= 255.0  # Adjust normalization if needed

    # Step 5: Feed the tensor through the face parsing CNN
    with torch.no_grad():
        output_tensor = model(tensor_image)
    #print('output',output_tensor.size())
    #print(output_tensor)
    # Step 6: Select the channel with the highest confidence as the segmentation mask
    segmentation_mask = torch.argmax(output_tensor, dim=1).squeeze(0)
    #print('dimension', segmentation_mask.size())

    # Step 7: Threshold the segmentation mask
    thresholded_mask = (segmentation_mask > 0).type(torch.float32)

    # Step 8: Resize the segmentation mask to 424x424 pixels
    resized_mask = transforms.Resize((424, 424), Image.NEAREST)(thresholded_mask.unsqueeze(0)).squeeze()

    # Step 9: Pad the segmentation mask by 96 pixels from all sides
    padded_mask = nn.ZeroPad2d(96)(resized_mask.unsqueeze(0)).squeeze()
    #print('dimension', padded_mask.size())

    return padded_mask



def extract_eye_zone(segmentation_mask, eye_class_id):
    if segmentation_mask is None:
        return None

    # Convert the segmentation mask to a NumPy array
    segmentation_np = segmentation_mask.cpu().numpy()

    # Create a binary mask for the eye class
    eye_mask = (segmentation_np == eye_class_id).astype(np.uint8)
    print('eye mask',eye_mask)
    # Find contours of the eye mask
    contours, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the eyes
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        eye_zone = (x, y, x + w, y + h)
        return eye_zone
    else:
        # No contours found, return a sentinel value or handle accordingly
        return (-1, -1, -1, -1)

    
def expand_rectangles(r1, r2, V):
    # Ensure r1 and r2 are not None
    if r1 is None or r2 is None:
        return None, None

    # Extract coordinates from rectangles
    x1, y1, x2, y2 = r1
    x3, y3, x4, y4 = r2

    # Expand rectangles on all sides by V pixels
    expanded_r1 = (x1 - V, y1 - V, x2 + V, y2 + V)
    expanded_r2 = (x3 - V, y3 - V, x4 + V, y4 + V)

    return expanded_r1, expanded_r2

if __name__ == "__main__":
    image_path= '/home/jing/FIQA_repo/datasets/neutral_front/neutral_front/063_03.jpg'
    R1,R2,eye_center,chin_point,left_eye,right_eye,counters,parsing = evaluate(image_path, cp='/home/jing/FIQA_repo/face_parsing_PyTorch/res/cp/79999_iter.pth')
    L_eye = np.array(eye_center)
    L16 = np.array(chin_point)
    T = np.linalg.norm(eye_center-L16)
    print('distance: ',T)
    
    #height & width
    with Image.open(image_path) as img:
        height = img.size[1]
        width = img.size[0]
    B = height
    A = width
    print('B:', B)
    print('A:',A)
    D=T/B
    print('D: ' , D)
    QC=np.round(200*(1-sigmoid(np.abs(D-0.45),0,0.05))) # change is done by the file Late-FR-comment-on-CD3-29794-5-231.. from christoph
    print(' QC for head size:' ,QC)
    
    #IED
    X_L=left_eye[0]
    Y_L=left_eye[1]
    X_R=right_eye[0]
    Y_R=right_eye[1]
    
    firstpart= (X_L - X_R) **2
    secondpart = (Y_L - Y_R) **2
    #IED =  math.sqrt((left_eye_center_point[0] - right_eye_center_point[0]) ** 2 + (left_eye_center_point[1] - right_eye_center_point[1]) ** 2)
    IED=np.sqrt(firstpart + secondpart) #by default set1/cos = 1
    
    print("IED: ",IED)
    
    #compute the IED quality component
    IED_QC = round(100 * (sigmoid(IED,70,20)))
    print(f"IED_QC: {IED_QC}")
        
    
    #T chin-midpoint between eyes
    
    eyes_middle_point_X = (X_L + X_R)/2
    eyes_middle_point_Y = (Y_L +Y_R)/2
    eyes_middle_point = (eyes_middle_point_X,eyes_middle_point_Y)
    
    
    #upper & lower
    #get upper and lower points for both left and right eyes
    left_upper_eye,left_lower_eye=find_upper_lower(left_eye,counters)
    right_upper_eye, right_lower_eye = find_upper_lower(right_eye,counters)
    print("left upper lower:",left_upper_eye,left_lower_eye)
    print("right upper lower:",right_upper_eye,right_lower_eye)
    
    #DL &DR
    DL = np.abs(left_lower_eye[1] - left_upper_eye[1])
    DR = np.abs(right_upper_eye[1] - right_lower_eye[1])
    
    D_PAL = min(DL,DR)
    
    #eyes open
    
    w_eye= D_PAL / T
    print('w (eye openness): ',w_eye)
    # compute the quality component of eye openness
    eye_openness_QC = round(100 * sigmoid(w_eye, 0.02, 0.01))
    
    print('D PAL:',D_PAL)
    print(f"eye opennes QC: {eye_openness_QC}")
    
    #mouth closed
    
    #lip upper_lower position
    upper_lip_positions,counter_lip = extract_eye_positions(parsing,eye_class_id=12)
    print('lip upper',upper_lip_positions)
    lower_lip_positions,counter_lip = extract_eye_positions(parsing,eye_class_id=13)
    print('lower lip:',lower_lip_positions)
    
    D_Lmouth= np.linalg.norm(np.array(upper_lip_positions) - np.array(lower_lip_positions))
    
    w_mouth = D_Lmouth / T
    
    # compute mouth closed quality component
    mouth_closed_QC = round(100 * (1 - sigmoid(w_mouth, 0.2, 0.06)))

    print(f"DL_mouth: {D_Lmouth}")
    print(f"W_mouth: {w_mouth}")
    print(f" mouth_closed_QC: {mouth_closed_QC}")
    
    
    
# compute the leftward crop QC of the face in image
    leftward_crop_QC = round(100 * (sigmoid(right_eye[0]/IED, 0.9, 0.1)))
    print(f"leftward_crop_QC = {leftward_crop_QC}")

    # compute the rightward crop QC of the face in image
    rightward_crop_QC = round(100 * (sigmoid((A - left_eye[0])/IED, 0.9, 0.1)))
    print(f"rightward_crop_QC = {rightward_crop_QC}") 

    # compute the downward crop QC of the face in image
    downward_crop_QC = round(100 * (sigmoid(eyes_middle_point[1]/IED, 1.4, 0.1)))
    print(f"dwonward_crop_QC = {downward_crop_QC}")  

    # compute the upward crop QC of the face in image
    upward_crop_QC = round(100 * (sigmoid((B - eyes_middle_point[1])/IED, 1.8, 0.1)))
    print(f"upward_crop_QC = {upward_crop_QC}")     


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
    O = O + extract_eye_zone(M,eye_class_id=5)
    O=torch.tensor(O)
    print('O',O)
    print('M',M)
    #V - offset distance
    V = IED/20
    
    expanded_r1, expanded_r2 = expand_rectangles(R1, R2, V)
    E = expanded_r1 + expanded_r2
    E = torch.tensor(E)
    print('E:',E)
    #print('O:',O)
    if O is None:
        alpha = 1
    else:
        alpha = (torch.abs(E)-torch.abs(O))/torch.abs(E)
        alpha_mean = torch.mean(alpha)
    #alpha_np = alpha.numpy()
    print('alpha:',alpha)
    
    #eye visible QC
    QC_EV = torch.round(100 * alpha_mean).item()
    print('Eye visible QC: ',QC_EV)



