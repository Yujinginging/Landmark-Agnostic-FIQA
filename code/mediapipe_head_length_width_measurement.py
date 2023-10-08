import cv2
import mediapipe as mp



#mediapipe face detection components:
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


#load the image: 
#one image for now. should be changed
image_path = '/home/jing/Landmark-Agnostic-FIQA/img_test/518.jpg'
image = cv2.imread(image_path)
#cv2.imshow('Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#face detection initialization:
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#face bounding box:
for detection in results.detections:
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    head_length = h
    head_width = w

    # Draw the bounding box on the image (optional)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #save the image
    cv2.imwrite('/home/jing/Landmark-Agnostic-FIQA/img_test/test_output/518_output_mediapipe.jpg',image)

    # Display head width and length
    print(f"Head Width: {head_width} pixels")
    print(f"Head Length: {head_length} pixels")

cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

