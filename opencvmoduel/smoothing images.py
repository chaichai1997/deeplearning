# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test3.JPG")
# 5*5矩阵 均值 2D convolution（image filter）图像滤波
"""
filter2D
    brief Convolves an image with the kernel
"""
# kernel = np.ones((5, 5), np.float32)/25
# dst = cv2.filter2D(img, -1, kernel)
#
# plt.subplot(121)
# plt.imshow(img)
# plt.title("Original")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(dst)
# plt.title("Averaging")
# plt.xticks([])
# plt.yticks([])
# plt.show()

# image Blurring(Smoothing)

# blur1 = cv2.blur(img, (5, 5))  # AVERAGE
# blur2 = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian Blurring
# median = cv2.medianBlur(img, 5)  # Median 中值模糊
# plt.subplot(221)
# plt.imshow(img)
# plt.title("Original")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(222)
# plt.imshow(blur1)
# plt.title("Average")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(223)
# plt.imshow(blur2)
# plt.title("Gaussian")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(224)
# plt.imshow(median)
# plt.title("Median")
# plt.xticks([])
# plt.yticks([])
# plt.show()

# Bilateral Filtering 双边滤波 可消除噪音并保持边缘
blur = cv2.bilateralFilter(img, 9, 75, 75)
plt.subplot(121)
plt.imshow(img)
plt.title("before")
plt.subplot(122)
plt.imshow(blur)
plt.title("after")
plt.show()



