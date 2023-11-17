from retinaface import RetinaFace
import cv2
import math


#load and preprocess image:
#now use just one, should be modified later:
image_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/518.jpg'
image = cv2.imread(image_path)

#face detection
faces = RetinaFace.detect_faces(image) #list of faces as json object where each represents a detected face.

#head length and width measurement
for face, value in faces.items():
    for cord,values in value.items():
        if cord == "facial_area": 
            x, y, w, h = values #the bounding box coordinates
            head_length = h - y  # calculate face length
            head_width = w - x # calculate face Width 

            # calculate the the  chin coordinates 
            chin_x = w  // 2
            chin_y = h 
            chin_coordinate = (round(chin_x), round(chin_y)) 

            # Draw the rectangle on the image (for visualization only)
            #cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    
            # Print the head width and length in pixels
            print(f"Head Width: {head_width} pixels")
            #print(f"Head Length: {head_length} pixels")
    
            #save the image with rectangles: for test and view results later
            #cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)
        if cord == "landmarks":
            eye1_x, eye1_y = values['left_eye'] # left eye coordinates
            eye2_x, eye2_y = values['right_eye'] # right eye coordinates

            #calculate the center coordinates between the eyes
            center_x = (eye1_x + eye2_x) // 2
            center_y = (eye1_y + eye2_y) // 2
            center_eye_coordinate = (round(center_x), round(center_y))

            # claculate the distance between the chin and the calculated centeral dot coordinates between the eyes
            distance = math.sqrt((center_eye_coordinate[0] - chin_coordinate[0]) ** 2 + (center_eye_coordinate[1] - chin_coordinate[1]) ** 2)

            # calculate the head length from the crown to the chin 
            head_length_from_crown = 2 * distance

            # Draw the line between the center of the eyes and the chain
            head_crown_y = round(center_y) -  round(distance) # claculate the crown y coordinates
            cv2.line(image, (center_eye_coordinate[0], head_crown_y), (center_eye_coordinate[0], chin_coordinate[1]), (0, 255, 0), 2)

            # Draw the line width of the head
            x1 = round(x )
            y = round(h // 2)
            x2 = round(w)
            cv2.line(image, (x1,y), (x2, y), (0, 255, 0), 2)

            # Print the head width and length in pixels
            print(f"Head Length from crown: {round(head_length_from_crown)} pixels")

            #save the image with rectangles: for test and view results later
            cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)


# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()



