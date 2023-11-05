from mtcnn import MTCNN
import cv2


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

    # Draw the rectangle on the image (for visualization only)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Print the head width and length in pixels
    print(f"Head Width: {head_width} pixels")
    print(f"Head Length: {head_length} pixels")
    
    #save the image with rectangles: for test and view results later
    cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)
    

# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()



