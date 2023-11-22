import urllib.request
import os
import dlib
import bz2
import shutil
import math
import cv2
import os
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

# Rest of your script...


# Load the shape predictor
predictor = dlib.shape_predictor(predictor_path)


#load from the folder
def get_images_from_folder(folder_path, extensions=['jpg', 'jpeg', 'png']):
   
    images = []
    for extension in extensions:
        pattern = os.path.join(folder_path, f'*.{extension}')
        images.extend(glob.glob(pattern))
    return images



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

    # Extract eye and chin points
        left_eye = landmarks[37]  # Left eye outer corner
        right_eye = landmarks[46]  # Rightg eye outer corner
        chin = landmarks[8]  # Chin
        left_side_head = landmarks[1]
        right_side_head = landmarks[15]

    # Calculate distances
        distance_left_eye_chin = math.sqrt((left_eye[0] - chin[0])**2 + (left_eye[1] - chin[1])**2)
        distance_right_eye_chin = math.sqrt((right_eye[0] - chin[0])**2 + (right_eye[1] - chin[1])**2)
    # Calculate the center point between left and right eyes
        center_point = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    #distance_center eye to chin
        distance_centereye_chin = chin[1] - center_point[1]
    #estimated point for head top
        top_point = (center_point[0], center_point[1] - distance_centereye_chin)
    #test the distance
        head_length_estimation = chin[1] - top_point[1] 
    #head width
        head_width = math.sqrt( (left_side_head[0] - right_side_head[0])**2 + (left_side_head[1] -right_side_head[1]) **2)
    
    # Print the distances
        print(f"Distance from eye center to chin: {distance_centereye_chin} pixels")
        print(f"head length estimation: {head_length_estimation} pixels")
        print(f"head width estimation: {head_width} pixels")



    # Draw the rectangle around the face (for visualization only)
    # x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw circles for eyes and chin (for visualization only)
        for (x, y) in [left_eye, right_eye, chin]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    #draw the lines
        cv2.line(image,left_side_head,right_side_head,(0,255,0),2)
        cv2.line(image, top_point, chin, (0,255,0), 2)

# Save the image with rectangles and landmarks
    output_path = image_path.replace('.jpg', '_dlib_output.jpg')
    cv2.imwrite(output_path, image)
    print(f"Image saved with rectangles and landmarks: {output_path}")


# Load and preprocess image
# image_path = '/home/jing/FIQA_repo/img_test/518_2.jpg'
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
folder_path = '/home/jing/FIQA_repo/img_test/resize'
image_files = get_images_from_folder(folder_path)

for image_file in image_files:
    get_dlib_output(image_file)