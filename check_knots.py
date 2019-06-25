import yaml
import os

def check(idx, knots_info):
	pixels = knots_info[idx]
	for (u, v) in pixels:
		if u > 640 or v > 480:
			print("out of bounds", idx, (u, v))

if __name__ == '__main__':
	with open("images/knots_info.yaml", "r") as stream:
		knots_info = yaml.safe_load(stream)
	for i in range(len(os.listdir('/Users/priyasundaresan/Desktop/rope-rendering/images')) - 1):
		check(i, knots_info)
