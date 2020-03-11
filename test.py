import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

filename = './rope_400_rgb_planar_test/processed/image_masks/000000_visible_mask.png'
img = cv2.imread(filename,0)
cv2.imshow("img", img)
cv2.waitKey(0)

img_tensor = torch.from_numpy(img)
print(img_tensor.shape)
img_tensor = img_tensor.type(torch.DoubleTensor) 
img_tensor.unsqueeze_(0)
img_tensor = img_tensor.view(1,1,480,640)
H,W = 28,28
new_img = F.interpolate(img_tensor, size=(H,W), mode='nearest')
cv_new_img = new_img.squeeze().numpy().astype(np.uint8)
cv_new_img = cv_new_img.reshape(H,W)
cv2.imshow("new", cv_new_img)
cv2.waitKey(0)
cv2.imshow("new", cv2.resize(cv_new_img, (640,480)))
cv2.waitKey(0)
