import cv2
import numpy as np
import os
import argparse

def mask(image_filename):
	img = cv2.imread('./images/{}'.format(image_filename))
	img[np.where((img == [255, 255, 255]).all(axis = 2))] = [0, 0, 0]
	img[np.where((img != [0, 0, 0]).all(axis = 2))] = [1, 1, 1]
	mask = img
	visible_mask = img * 255
	mask_filename = image_filename.replace('rgb', 'mask')
	visible_mask_filename = image_filename.replace('rgb', 'visible_mask')
	cv2.imwrite('image_masks/{}'.format(mask_filename), mask)
	cv2.imwrite('image_masks/{}'.format(visible_mask_filename), visible_mask)


if __name__ == '__main__':
	for filename in os.listdir('./images'):
		try:
			print("Masking %s" % filename)
			mask(filename)
		except:
			print("Done")
