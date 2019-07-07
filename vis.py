import cv2
import numpy as np
import argparse
import yaml
import os
import math
import json

def show_knots(idx, knots_info, save=True):
	image_filename = "{0:06d}_rgb.png".format(idx)
	print(image_filename)
	img = cv2.imread('images/{}'.format(image_filename))
	pixels = knots_info[str(idx)]
	for i in range(len(pixels)):
		# if pixels[i][1]:
		valid = pixels[i][1]
		(u, v) = pixels[i][0]
		val = 255 * i/len(pixels)
		if valid:
			color = (val, 255 - val, 255)
		else:
			color = (255, 255, 255)
		cv2.circle(img,(int(u), int(v)), 1, color, -1)
	if save:
		annotated_filename = "{0:06d}_annotated.png".format(idx)
		cv2.imwrite('./annotated/{}'.format(annotated_filename), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=len(os.listdir('./images')) - 1)
    args = parser.parse_args()
    print("parsed")
    # with open("images/knots_info.yaml", "r") as stream:
	   #  knots_info = yaml.safe_load(stream)
    with open("images/knots_info.json", "r") as stream:
    	knots_info = json.load(stream)
    print("loaded knots info")
    for i in range(args.num):
	    show_knots(i, knots_info)
