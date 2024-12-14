import torchvision.utils as vutils
import argparse
import tqdm
import cv2
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Create a blurred dataset from the original dataset.")
parser.add_argument('--datasize', type=int, default=63560, help="Total size of the dataset.")
parser.add_argument('--blurratio', type=float, default=0.1, help="Proportion of the dataset to blur (0-1).")
parser.add_argument('--data_path', type=str, default='/root/BaMSGAN_GitHub/data/data_original/class', help="Path to the original image dataset.")
parser.add_argument('--save_path', type=str, default='/root/BaMSGAN_GitHub/data/data_blur/class', help="Path to save the blurred images.")
parser.add_argument('--t1', type=int, default=200, help="Lower threshold for Canny edge detection.")
parser.add_argument('--t2', type=int, default=500, help="Upper threshold for Canny edge detection.")
opt = parser.parse_args()

# Image blur processing function
def Blur_process(img_path, ts1, ts2):
    img = cv2.imread(img_path)
    img_copy = img.copy()
    imgCanny = cv2.Canny(img, ts1, ts2)
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    img_dilate = cv2.dilate(imgCanny, g)
    img_dilate2 = cv2.dilate(imgCanny, g2)

    # Use NumPy array operations for performance
    img[imgCanny == 0] = [0, 0, 0]  # Black out non-edge regions
    dst = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    img_copy[imgCanny != 0] = dst[imgCanny != 0]  # Apply blur to non-edge areas
    return img_copy

# Main processing logic
if __name__ == "__main__":
    # Create save path if it does not exist
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # Image processing
    i = 0
    for file in tqdm.tqdm(os.listdir(opt.data_path), desc="Processing images"):
        img_path = os.path.join(opt.data_path, file)
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {file}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read file {file}. Skipping...")
            continue
        
        img_copy = Blur_process(img_path, opt.t1, opt.t2)
        name = f'blurred_{i}.jpg'
        cv2.imwrite(os.path.join(opt.save_path, name), img_copy)
        i += 1
        if i >= opt.datasize * opt.blurratio:
            print("Reached the target number of blurred images. Stopping...")
            break