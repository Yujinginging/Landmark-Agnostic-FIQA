import sys
sys.path.append('/home/jing/FIQA_repo/face_parsing.PyTorch') 
from model import BiSeNet
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.jpg', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(image_path, cp='cp/79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        return parsing

def get_head_measurements(image_path, cp):
    # Load the parsing result using the provided code
    parsing_result = evaluate(image_path=image_path, cp=cp)

    # Define the class ID for the face in the parsing result
    face_class_id = 1

    # Generate a face mask
    face_mask = (parsing_result == face_class_id).astype(np.uint8)

    # Find contours in the face mask
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Assuming the largest contour corresponds to the face
        largest_contour = max(contours, key=cv2.contourArea)
        bounding_box = cv2.boundingRect(largest_contour)
        
        # Adjust the bounding box to be lower, closer to the chin area
        bounding_box = (bounding_box[0] + int(bounding_box[3] / 4), bounding_box[1] + int(bounding_box[3]),
                        int(bounding_box[2] * 2.2), int(4 * bounding_box[3] / 3 - 6))

        # Draw bounding box on the original image
        img = cv2.imread(image_path)
        # cv2.rectangle(img, (bounding_box[0], bounding_box[1]),
        #               (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
        #               (0, 255, 0), 2)  # Green bounding box

        # Draw lines for head length and width
        head_length = 2 * bounding_box[3]
        head_width = bounding_box[2]

        # Line for head length
        cv2.line(img, (bounding_box[0] + int(bounding_box[2]/2) , bounding_box[1] - head_length),
                 (bounding_box[0] + int(bounding_box[2]/2), bounding_box[1] + bounding_box[3]), (255, 0, 0), 2)

        # Line for head width
        cv2.line(img, (bounding_box[0], bounding_box[1] + int(bounding_box[3])//4),
                 (bounding_box[0] + int(head_width), bounding_box[1] + int(bounding_box[3])//4), (0, 0, 255), 2)

        # Save the image with the bounding box and lines
        output_image_path = image_path.replace('.jpg', '_faceparsing.jpg')
        cv2.imwrite(output_image_path, img)

        print("The estimation of the head length is:", head_length, "pixels")
        print("The estimation of the head width is:", head_width, "pixels")

        return output_image_path

    else:
        print("No face contour found.")
        return None

if __name__ == "__main__":
    image_path = '/home/jing/FIQA_repo/face_parsing.PyTorch/res/img/568_2.jpg'  
    cp = '/home/jing/FIQA_repo/face_parsing.PyTorch/res/cp/79999_iter.pth'   

    output_image_path = get_head_measurements(image_path, cp)

    if output_image_path is not None:
        print(f"Output image saved at: {output_image_path}")
