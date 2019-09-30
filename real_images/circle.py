import numpy as np
import cv2

for i in range(20):
    img = cv2.imread('segdepth_%d.png' % i, 0)
    output = img.copy()
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=9,minRadius=35,maxRadius=50)
    if circles is not None:
        for i in circles[0,:]:
            i = [int(i[0]), int(i[1]), int(i[2])]
            if img[i[1], i[0]] != 0:
                cv2.circle(output,(i[0],i[1]),i[2],(255,255,255),2)
                cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)
                print(i[0], i[1])
                break
    cv2.imshow("circled", output)
    cv2.waitKey(0)
