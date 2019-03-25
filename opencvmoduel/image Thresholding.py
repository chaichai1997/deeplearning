# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
# image thresholiding 图像阈值化
img = cv2.imread("test.jpg", 0)
# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.CV_8U)

# Simple Thresholding简单阈值

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

# ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# Adaptive Thresholding自适应阈值
"""
It has three ‘special’ input params and only one output argument.
    Adaptive Method:
        cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood 
values where weights are a gaussian window.
    Block size: It decides the size of neighbourhood area.
    C - It is just a constant which is subtracted from the mean or weighted mean calculated.
cv2.AdaptiveThreshold
    .   @param src Source 8-bit single-channel image.
    .   @param dst Destination image of the same size and the same type as src.
    .   @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied
    .   @param adaptiveMethod Adaptive thresholding algorithm to use, see #AdaptiveThresholdTypes.
    .   The #BORDER_REPLICATE | #BORDER_ISOLATED is used to process boundaries.
    .   @param thresholdType Thresholding type that must be either #THRESH_BINARY or #THRESH_BINARY_INV,
    .   see #ThresholdTypes.
    .   @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the
    .   pixel: 3, 5, 7, and so on.
    .   @param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it
    .   is positive but may be zero or negative as well.
"""

# gray1 = cv2.medianBlur(img, 5)  # brief Blurs an image using the median filter使用中值滤波模糊图像
# ret, th1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                             cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                             cv2.THRESH_BINARY, 11, 2)
# titles = ["Original", "Global Thresholding(v=127)",
#           "Adaptive Thresholding", "Adaptive Gaussian thresholding"]
# images = [gray1, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# Ostu's Binarization
