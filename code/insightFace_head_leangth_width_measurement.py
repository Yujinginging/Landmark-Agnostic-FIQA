import cv2
from insightface.app import FaceAnalysis

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
head_width= abs(face.bbox[2] - face.bbox[0])
head_length= abs(face.bbox[3] - face.bbox[1]) 

print(f"Head Width: {head_width} pixels")
print(f"Head Length: {head_length} pixels")

# Draw the rectangle on the image (for visualization only)
cv2.rectangle(image, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]),int(face.bbox[3])), (0, 255, 0), 2)

#save the image with rectangles: for test and view results later
cv2.imwrite('/home/teakoo/Landmark-Agnostic-FIQA/img_test/test_output/518_withRect.jpg',image)

# Display or save the image with rectangles (optional)
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
