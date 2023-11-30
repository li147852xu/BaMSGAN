import argparse
import cv2
import torchvision.utils as vutils
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasize', type=int, default=63560)
parser.add_argument('--blurratio', type=int, default=0.1)
opt = parser.parse_args()

def Preview(img_path, ts1, ts2):
    img = cv2.imread(img_path)
    img_copy = img.copy()
    imgCanny = cv2.Canny(img, ts1, ts2)
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    img_dilate = cv2.dilate(imgCanny, g)
    img_dilate2 = cv2.dilate(imgCanny, g2)

    shape = img_dilate.shape
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if imgCanny[i, j] == 0:
                img[i, j] = [0, 0, 0] 
    
    dst = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if imgCanny[i, j] != 0:
                img_copy[i, j] = dst[i, j]
    return img_copy
    

t1 = 200
t2 = 500
data_path = '/root/BaMSGAN_GitHub/data/images'
i = 0
save_path = '/root/BaMSGAN_GitHub/data_blur/Blur_images'
for file in tqdm.tqdm(os.listdir(data_path)):
    img_path = os.path.join(data_path, file)
    img = cv2.imread(file)
    img_copy = Preview(img_path, t1, t2)
    name = 'blurred' + str(i) + '.jpg'
    cv2.imwrite(os.path.join(save_path, name), img_copy)
    i += 1
    if i == opt.datasize * opt.blurratio:
        break
