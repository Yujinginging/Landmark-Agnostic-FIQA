from mtcnn import MTCNN
import cv2
import math


#load MTCNN detector
detector = MTCNN() #pre_trained face detector

#load and preprocess image:
#now use just one, should be modified later:
image_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/518.jpg'
image = cv2.imread(image_path)

#face detection
faces = detector.detect_faces(image) #list of rectangles where each represents a detected face.

#head length and width measurement
for face in faces:
    x, y, w, h = face['box'] #x,y:top-left corner coordinates
    head_length = h  # Height of the rectangle is considered head length
    head_width = w  # Width of the rectangle is considered head width
    eye1_x, eye1_y = face['keypoints']['left_eye'] # left eye coordinates
    eye2_x, eye2_y = face['keypoints']['right_eye'] # right eye coordinates

    #calculate the center coordinates between the eyes
    center_x = (eye1_x + eye2_x) // 2
    center_y = (eye1_y + eye2_y) // 2
    center_eye_coordinate = (center_x, center_y)

    # calculate the the  chin coordinates 
    chin_x = x + w // 2
    chin_y = y + h 
    chin_coordinate = (chin_x, chin_y) 

    # claculate the distance between the chin and the calculated centeral dot coordinates between the eyes
    distance = math.sqrt((center_eye_coordinate[0] - chin_coordinate[0]) ** 2 + (center_eye_coordinate[1] - chin_coordinate[1]) ** 2)

    # calculate the head length from the crown to the chin 
    head_length_from_crown = 2 * distance

    # Draw the rectangle on the image (for visualization only)
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw the line between the center of the eyes and the chain
    head_crown_y = (center_y) -  round(distance) # claculate the crown y coordinates
    cv2.line(image, (chin_x, head_crown_y), (chin_x, chin_y), (0, 255, 0), 2)
    
    # Print the head width and length in pixels
    print(f"Head Width: {head_width} pixels")
    print(f"Head Length: {head_length} pixels")
    print(f"Head Length from crown: {head_length_from_crown} pixels")

    
    #save the image with rectangles: for test and view results later
    cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)
    

# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()



