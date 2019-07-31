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
    img2 = img.copy()
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
    _, mask = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(60, 60))
    img2 = clahe.apply(img2)
    img2 = cv2.bitwise_not(img2, mask=mask)
    #s_min = np.amin(img2[np.where((img2!=[np.amin(img2)]))])
    #s_max = np.amax(img2[np.where((img2!=[np.amax(img2)]))])
    #t_min = np.amin(template[np.where((template!=[np.amin(template)]))])
    #t_max = np.amax(template[np.where((template!=[np.amax(template)]))])
    img2[np.where((img2 != [0]))] += 25
    #img2 = cv2.bilateralFilter(img2,9,75,75)
#    m = interp1d([s_max, s_min],[t_min, t_max])
#    for val in range(s_min, s_max):
#        img2[np.where((img2==[val]))] = [m(val).squeeze()]
        
    #cv2.imwrite("mask.png", template)
    #matched = match_histograms(img2, template, multichannel=False)
#    matched[np.where((matched == np.amin(matched)))] = 0
    cv2.imwrite('./{}/output/'.format(args.dir) + f, img2)
#cv2.imshow('dst', img2)                          
#cv2.waitKey(0)                                  
#cv2.destroyAllWindows()
    

