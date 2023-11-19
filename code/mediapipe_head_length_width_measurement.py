import cv2
import mediapipe as mp
import math

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Read an image
image_path = '/home/jing/FIQA_repo/img_test/518_2.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with Mediapipe FaceMesh
results = face_mesh.process(image_rgb)

# Extract the landmarks
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark

    # Indices for the center of the eyes and chin
    left_eye_index = 173
    right_eye_index = 398
    chin_index = 152
    left_jaw_index = 234
    right_jaw_index = 454

    # Get the coordinates of the center of the eyes and chin
    left_eye_point = (int(landmarks[left_eye_index].x * image.shape[1]), int(landmarks[left_eye_index].y * image.shape[0]))
    right_eye_point = (int(landmarks[right_eye_index].x * image.shape[1]), int(landmarks[right_eye_index].y * image.shape[0]))
    
    chin_point = (int(landmarks[chin_index].x * image.shape[1]), int(landmarks[chin_index].y * image.shape[0]))
    center_eye_point = ((left_eye_point[0] + right_eye_point[0]) // 2, (left_eye_point[1] + right_eye_point[1]) // 2)
    
    #coordinates of the jaw points
    left_jaw_point = (int(landmarks[left_jaw_index].x * image.shape[1]), int(landmarks[left_jaw_index].y * image.shape[0]))

    right_jaw_point = (int(landmarks[right_jaw_index].x * image.shape[1]), int(landmarks[right_jaw_index].y * image.shape[0]))

    # Calculate the distance between the center of the eyes and the chin
    eye_chin_distance = chin_point[1] - center_eye_point[1]
    
    #top head point estimation
    top_head_estimation = (center_eye_point[0], (center_eye_point[1]-eye_chin_distance))

    # Estimate head length as twice the distance between the center of the eyes and the chin
    head_length = 2 * eye_chin_distance
   

    # Calculate the head width as the distance between the first and last points of the jawline
    head_width = math.sqrt((left_jaw_point[0] - right_jaw_point[0])**2 + (left_jaw_point[1] - right_jaw_point[1])**2)

    print(f"Head Length estimation: {head_length} pixels")
    print(f"Head Width estimation: {head_width} pixels")

    # Draw the points and lines on the image for visualization
    
    cv2.circle(image, chin_point, 3, (0, 255, 0), -1)

    # Draw lines representing the distances
    cv2.line(image, top_head_estimation, chin_point, (0, 0, 255), 2)
    cv2.line(image, left_jaw_point, right_jaw_point, (0, 0, 255), 2)

    

    # Save the output image
    output_path = '/home/jing/FIQA_repo/img_test/test_output/518_2_mediapipe.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output image saved at: {output_path}")


else:
    print("No face detected in the image.")
