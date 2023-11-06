from retinaface import RetinaFace
import cv2


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

            # Draw the rectangle on the image (for visualization only)
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    
            # Print the head width and length in pixels
            print(f"Head Width: {head_width} pixels")
            print(f"Head Length: {head_length} pixels")
    
            #save the image with rectangles: for test and view results later
            cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)
    

# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()



