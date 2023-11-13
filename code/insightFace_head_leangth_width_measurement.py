import cv2
from insightface.app import FaceAnalysis
import math

#load and preprocess image:
#now use just one, should be modified later:
image_path = '/home/teakoo/face-parsing.PyTorch/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/7.jpg'
image = cv2.imread(image_path)

# load the model 
detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
detector.prepare(ctx_id=0, det_size=(640, 640))

# faces detection
faces = detector.get(image)

#head length and width measurement
face = faces[0]
head_width= abs(face.bbox[2] - face.bbox[0]) # calculate face width
head_length= abs(face.bbox[3] - face.bbox[1]) # calculate face length
eye1_x, eye1_y = face.kps[0] # left eye coordinates
eye2_x, eye2_y = face.kps[1] # right eye coordinates

#calculate the center coordinates between the eyes
center_x = (eye1_x + eye2_x) // 2
center_y = (eye1_y + eye2_y) // 2
center_eye_coordinate = (round(center_x), round(center_y))

# calculate the the  chin coordinates 
chin_x = face.bbox[2] // 2
chin_y = face.bbox[3]
chin_coordinate = (round(chin_x), round(chin_y)) 

# claculate the distance between the chin and the calculated centeral dot coordinates between the eyes
distance = math.sqrt((center_eye_coordinate[0] - chin_coordinate[0]) ** 2 + (center_eye_coordinate[1] - chin_coordinate[1]) ** 2)

# calculate the head length from the crown to the chin 
head_length_from_crown = 2 * distance

# Draw the line between the center of the eyes and the chain
head_crown_y = round(center_y) -  round(distance) # claculate the crown y coordinates
cv2.line(image, (center_eye_coordinate[0], head_crown_y), (center_eye_coordinate[0], chin_coordinate[1]), (0, 255, 0), 2)

# Print the head width and length in pixels
print(f"Head Length from crown: {round(head_length_from_crown)} pixels")

#save the image with rectangles: for test and view results later
cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)

print(f"Head Width: {head_width} pixels")
print(f"Head Length: {head_length} pixels")

# Draw the rectangle on the image (for visualization only)
#cv2.rectangle(image, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]),int(face.bbox[3])), (0, 255, 0), 2)

#save the image with rectangles: for test and view results later
#cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)

# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
