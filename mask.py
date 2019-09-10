import cv2
import numpy as np
import os
import argparse

def mask(image_filename, directory):
    ''' Produces a mask of a depth image by thresholding '''
    img_original = cv2.imread('./%s/%s'%(directory, image_filename))
    img = img_original.copy()
    #img[np.where((img > [250, 250, 250]).all(axis = 2))] = [0, 0, 0]
    img[np.where((img < [5, 5, 5]).all(axis = 2))] = [0, 0, 0]
    img[np.where((img != [0, 0, 0]).all(axis = 2))] = [1, 1, 1]
    mask = img
    visible_mask = img * 255
    mask_filename = image_filename.replace('rgb', 'mask')
    visible_mask_filename = image_filename.replace('rgb', 'visible_mask')
    cv2.imwrite('image_masks/{}'.format(mask_filename), mask)
    cv2.imwrite('image_masks/{}'.format(visible_mask_filename), visible_mask)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='images')
	args = parser.parse_args()
	if not os.path.exists("./image_masks"):
		os.makedirs('./image_masks')
	else:
		os.system('rm -rf ./image_masks')
		os.makedirs('./image_masks')
	for filename in os.listdir('./{}'.format(args.dir)):
		try:
			print("Masking %s" % filename)
			mask(filename, args.dir)
		except:
			print("Done")
