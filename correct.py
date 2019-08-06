import cv2
import numpy as np
import os
import argparse
from skimage.transform import match_histograms
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()
template = cv2.imread('./images/000000_rgb.png', 0)
if not os.path.exists('%s/output' % args.dir):
    os.makedirs('%s/output' % args.dir)
for f in os.listdir('./{}'.format(args.dir)):
    if not 'png' in f:
        continue
    img = cv2.imread('./{}/'.format(args.dir) + f, 0)
    #img2 = img.copy()
    #img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)) # close artifacts/black spots
    #_, mask = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(60, 60)) # bring back texture to depth img, contrast adaptive hist eq.
    #img2 = clahe.apply(img2)
    #img2 = cv2.bitwise_not(img2, mask=mask) # the depth in real is opposite sim (closer = darker), adjusts for this
    #s_med = np.mean(img2[np.where((img2>[50]))]) # get the mean pixel value of the real rope
    #t_med = np.mean(template[np.where((template>[50]))]) # get the mean pixel value of sim rope
    ##img2[np.where((img2 != [0]))] += int(t_med) - int(s_med) # adjust real range to better match sim
    #img2[np.where((img2 != [0]))] += 70 # adjust real range to better match sim
#    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
    img2 = img
    img2 = cv2.resize(img2,(640,480))
    cv2.imwrite('./{}/output/'.format(args.dir) + f, img2)
    

