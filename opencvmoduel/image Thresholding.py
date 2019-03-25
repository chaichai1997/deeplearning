# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
# image thresholiding 图像阈值化
img = cv2.imread("test.jpg", 0)
# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.CV_8U)
"""
def threshold(src, thresh, maxval, type, dst=None):
        @param src input array (multiple-channel, 8-bit or 32-bit floating point).
    .   @param dst output array of the same size  and type and the same number of channels as src.
    .   @param thresh threshold value.
    .   @param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding
    .   types.
    .   @param type thresholding type (see #ThresholdTypes).
    .   @return the computed threshold value if Otsu's or Triangle methods used.
"""
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()