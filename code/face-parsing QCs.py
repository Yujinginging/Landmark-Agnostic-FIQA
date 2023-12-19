from logger import setup_logger
from model import BiSeNet

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

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
        cv2.circle(vis_image, (cx, cy), 5, (0, 255, 0), 5)  # Green circle for eye positions

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
    image = img.resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    out = net(img)[0]
    parsing = out.detach().squeeze(0).cpu().numpy().argmax(0)

    # Extract positions of left and right eyes 
    left_eye_positions,counters = extract_eye_positions(parsing, eye_class_id=4)
    # a pair of positions[left,right]
    
    #chin
    chin_point = extract_chin_position(parsing,chin_class_id=1)
    print ('chin:',chin_point)
    
   
    
    # Visualize eye positions on the image
    #visualize_eye_positions(np.array(image), left_eye_positions)
    
    # Calculate the center positions
    left_eye_center = tuple(np.round(np.mean(left_eye_positions, axis=0)).astype(int))
    print('Left Eye Center:', left_eye_center)
    
    
    

   
    
    print(left_eye_positions)
    
    return left_eye_center,chin_point,left_eye_positions[0],left_eye_positions[1],counters,parsing

def sigmoid(x,a,b):
    return 1 / (1 + np.exp(-(x - a) / b))



if __name__ == "__main__":
    image_path='/home/jing/FIQA_repo/imgs/qualitycomponent/img/006_03.jpg'
    eye_center,chin_point,left_eye,right_eye,counters,parsing = evaluate(image_path, cp='/home/jing/FIQA_repo/face_parsing.PyTorch/res/cp/79999_iter.pth')
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
    QC=np.round(100*(1-sigmoid(np.abs(D-75),0,5)))
    print(' QC for head size:' ,QC)
    
    #IED
    X_L=left_eye[0]
    Y_L=left_eye[1]
    X_R=right_eye[0]
    Y_R=right_eye[1]
    
    firstpart= (X_L - X_R) **2
    secondpart = (Y_L - Y_R) **2
    IED=np.sqrt(firstpart + secondpart) #by default set1/cos = 1
    
    print("IED: ",IED)
    
    
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






