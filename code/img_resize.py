# Import the Images module from pillow 
from PIL import Image
import cv2 

# Open the image by specifying the image path. 
image_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/518.jpg'
image_file = Image.open(image_path)

# get the width and height of picture
width, height = Image.open(image_path).size
width1 = round(width * 3/4 )
height1 = round(height * 3/4)
new_image = image_file.resize((width1, height1))
new_image.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_1.jpg')
img1 =cv2.imread('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_1.jpg')
print('Image Width is',img1.shape[1])
print('Image Height is',img1.shape[0])
width2 = round(width * 1/2 )
height2 = round(height * 1/2)
new_image = image_file.resize((width2, height2))
new_image.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_2.jpg')
img1 =cv2.imread('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_2.jpg')
print('Image Width is',img1.shape[1])
print('Image Height is',img1.shape[0])
width3 = round(width * 1/4 )
height3 = round(height * 1/4)
new_image = image_file.resize((width3, height3))
new_image.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_3.jpg')
img1 =cv2.imread('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image518_3.jpg')
print('Image Width is',img1.shape[1])
print('Image Height is',img1.shape[0])

# Example-2 
#image_file.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image1_2.jpg', quality=10) 

