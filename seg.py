import numpy as np
import os
import cv2
from sklearn.neighbors import NearestNeighbors

def pixelNN(idxs, model, inputs, img):
    pixels = np.dstack(idxs[:2]).squeeze()
    closest_px = model.kneighbors(pixels, 1, return_distance=False).squeeze()
    matched_pixels = inputs[closest_px].squeeze()
    x = matched_pixels[:,0]
    y = matched_pixels[:,1]
    return img[x, y]

def segment(mask, color, annotated, i):
    #segmented = mask.copy()
    segmented = color.copy()
    colored_idxs = np.where((annotated > [1, 1, 1]))
    annotated_pixels = np.dstack(colored_idxs[:2]).squeeze()
    neigh = NearestNeighbors(1, 5)
    neigh.fit(annotated_pixels)
    masked_indices = np.where((mask == [255, 255, 255]))
    segmented[masked_indices[:2]] = [pixelNN(masked_indices, neigh, annotated_pixels, annotated)]
    cv2.imwrite('partitioned/%06d_segmented.png' % i, segmented)
    #indices = np.where((img <= [a]) & (img >= [b]))
    #img[indices] = [m(img[indices])]


if __name__=='__main__':
    if not os.path.exists("./partitioned"):
        os.makedirs('./partitioned')
    else:
        os.system('rm -rf ./partitioned')
        os.makedirs('./partitioned')
    for i in range(len(os.listdir('annotated'))):
        mask = cv2.imread('image_masks/%06d_visible_mask.png' % i)
        annotated = cv2.imread('annotated/%06d_annotated.png' % i)
        #color = cv2.imread('images/%06d_rgb.png' % i)
        color = cv2.imread('images/%06d.jpg' % i)
        print("Segmenting image_masks/%06d_visible_mask.png" % i)
        segment(mask, color, annotated, i)
