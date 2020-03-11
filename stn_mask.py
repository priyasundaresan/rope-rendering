import cv2
import numpy as np
import os
import argparse

def mask(image_filename, directory):
    ''' Produces a mask of a depth image by thresholding '''
    img_original = cv2.imread('./%s/%s'%(directory, image_filename), 0)
    _, mask = cv2.threshold(img_original, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (40, 40))
    mask_filename = image_filename.replace('rgb', 'mask')
    cv2.imwrite('image_masks/{}'.format(mask_filename), mask)

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
