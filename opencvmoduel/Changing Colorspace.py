# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
# opencv中颜色分类
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
# 对象追踪
cap = cv2.VideoCapture(0)
while(1):
    frame = cv2.imread("test3.JPG")
    # convert RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0, 50, 50])
    upper_blue = np.array([50, 255, 255])

    # Threshold the HSV image to get only blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise_AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    plt.subplot(131)
    plt.imshow(frame)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(res)
    plt.show()

# Take each frame

