# Import the Images module from pillow 
from PIL import Image
import cv2 

# Open the image by specifying the image path. 
image_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/dlib_testing/173_03.jpg'
image_file = Image.open(image_path)

# get the width and height of picture
width, height = Image.open(image_path).size
width1 = round(width * 1/2 )
height1 = round(height * 1/2)
new_image = image_file.resize((width1, height1))
new_image.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/173_03_1.jpg')
img1 =cv2.imread('/home/teakoo/Landmark-Agnostic-FIQA/img_test/173_03_1.jpg')


# Example-2 
#image_file.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image1_2.jpg', quality=10) 

from PIL import Image
import os
import cv2

def resize_and_save(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Construct the full path to the input image
            input_image_path = os.path.join(input_folder, filename)

            # Open the image
            image_file = Image.open(input_image_path)

            # Get the width and height of the picture
            width, height = image_file.size
            width1 = round(width * 1/2)
            height1 = round(height * 1/2)

            # Resize the image
            new_image = image_file.resize((width1, height1))

            # Construct the full path to the output image
            output_image_path = os.path.join(output_folder, filename)

            # Save the resized image
            new_image.save(output_image_path)

            # Display or process the resized image using OpenCV (optional)
            img = cv2.imread(output_image_path)
            # Perform further processing if needed

            # Print the paths for verification (you can remove this line)
            print(f"Input Image: {input_image_path}, Output Image: {output_image_path}")

# Example usage
input_folder_path = '/path/to/your/input/images/folder'
output_folder_path = '/path/to/your/output/images/folder'

resize_and_save(input_folder_path, output_folder_path)
