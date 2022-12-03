# -*- coding: UTF-8 -*-
"""
@Project : FT_Pro
@File    : 02ReFT_Demo.py
@IDE     : PyCharm 
@Author  : 李睡睡
@Date    : 2022/12/2 17:18 
@Scripts :opencv 、 numpy 、 matplotlib
@Documentation:
@Function:分别实现了 傅里叶变换的 正变换 与 逆变换
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)

# 傅里叶正变换 ： 图像 -> 复数
f = np.fft.fft2(img)

# fshift 为 低频转移到中心的值（复数）
fshift = np.fft.fftshift(f)

# ishift 为 低频在左上的值（复数）
ishift = np.fft.fftshift(fshift)

# 低频在左上的复数 进行 逆傅里叶变换 得到 iimg ，仍为复数
iimage = np.fft.ifft2(ishift)

# 使用abs转换为实数
iimage = np.abs(iimage)

plt.subplot(121)
plt.imshow(img,cmap='gray')

plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(iimage,cmap='gray')

plt.title('iimage')
plt.axis('off')

plt.savefig('../out/ReFT_lena')
plt.show()