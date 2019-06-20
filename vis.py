import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    pixels = np.loadtxt('pixels.txt')
    for (u, v) in pixels:
    	cv2.circle(img,(int(u), int(v)), 5, (255,255,0), -1)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
