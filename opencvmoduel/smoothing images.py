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
"""
average()
"""
blur = cv2.blur(img, (5, 5))
plt.subplot(121)
plt.imshow(img)
plt.title("Original")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(blur)
plt.title("Blurred")
plt.xticks([])
plt.yticks([])
plt.show()