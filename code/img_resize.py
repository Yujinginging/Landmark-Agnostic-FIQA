# Import the Images module from pillow 
from PIL import Image 

# Open the image by specifying the image path. 
image_path = '/home/teakoo/Landmark-Agnostic-FIQA/img_test/1.jpg'
image_file = Image.open(image_path) 

# the default 
image_file.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image1_0.jpg', quality=95) 

# Changing the image resolution using quality parameter 
# Example-1 
image_file.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image1_1.jpg', quality=25) 

# Example-2 
image_file.save('/home/teakoo/Landmark-Agnostic-FIQA/img_test/image1_2.jpg', quality=10) 
print('ok')
