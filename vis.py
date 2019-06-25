import cv2
import numpy as np
import argparse
import yaml
import os

def show_knots(idx, knots_info, save=True):
	image_filename = "{0:06d}_rgb.png".format(idx)
	print(image_filename)
	img = cv2.imread('images/{}'.format(image_filename))
	pixels = knots_info[idx]
	for (u, v) in pixels:
		cv2.circle(img,(int(u), int(v)), 5, (255,255,0), -1)
	if save:
		annotated_filename = "{0:06d}_annotated.png".format(idx)
		cv2.imwrite('annotated/{}'.format(annotated_filename), img)


if __name__ == '__main__':

	with open("images/knots_info.yaml", "r") as stream:
		knots_info = yaml.safe_load(stream)
	for i in range(len(os.listdir('/Users/priyasundaresan/Desktop/rope-rendering/images')) - 1):
		show_knots(i, knots_info)
