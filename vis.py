import cv2
import numpy as np
import argparse
import yaml
import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    filename = "{0:06d}_rgb.png".format(args.index)
    img = cv2.imread('images/{}'.format(filename))
    with open("images/knots_info.yaml", "r") as stream:
        knots_info = yaml.safe_load(stream)
        pprint.pprint(knots_info)
    pixels = knots_info[filename]
    for (u, v) in pixels:
    	cv2.circle(img,(int(u), int(v)), 5, (255,255,0), -1)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
