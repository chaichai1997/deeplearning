# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Morphological Transformations 形态转换
     different morphological operations
like Erosion, Dilation, Opening, Closing etc.
 腐蚀 膨胀 打开 闭合
"""
img = cv2.imread("test1.jpg")
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)  # erosion腐蚀
dilation = cv2.dilate(img, kernel, iterations=1)  # dilation膨胀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # opening 打开
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # closing 闭合
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # Morphological Gradient形态梯度
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # 图像与opening打开的区别
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # 图像与close闭合的区别
a = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]
titiles = ['img', 'erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(a[i])
    plt.title(titiles[i])
plt.show()


