# Import the Images module from pillow 
from PIL import Image 

# Open the image by specifying the image path. 
image_path = '/home/jing/FIQA_repo/face_parsing.PyTorch/res/img/568.jpg'
image_file = Image.open(image_path) 

# the default 
image_file.save('/home/jing/FIQA_repo/face_parsing.PyTorch/res/img/568_0.jpg', quality=95) 

# Changing the image resolution using quality parameter 
# Example-1 
image_file.save('/home/jing/FIQA_repo/face_parsing.PyTorch/res/img/568_1.jpg', quality=25) 

# Example-2 
image_file.save('/home/jing/FIQA_repo/face_parsing.PyTorch/res/img/568_2.jpg', quality=10) 
print('ok')