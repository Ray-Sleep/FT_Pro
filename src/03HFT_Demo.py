# -*- coding: UTF-8 -*-
"""
@Project : FT_Pro
@File    : 03HFT_Demo
@IDE     : PyCharm
@Author  : 李睡睡
@Date    : 2022/12/4 2:32
@Scripts :opencv 、 numpy 、 matplotlib
@Documentation:https://www.bilibili.com/video/BV1qt4y1u7N9?p=6&vd_source=fa4736bfe6ad0afc755decfbf72b082a
@Function:使用 numpy 对高通滤波的实现
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 设置高通滤波，(去掉低频)
rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)
fshift[crow-30 : crow+30,ccol-30:ccol+30] = 0

# 逆傅里叶变换
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

# 显示原始图像
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original'),plt.axis('off')

# 显示滤波结果
plt.subplot(122),plt.imshow(iimg,cmap='gray')
plt.title('iimg'),plt.axis('off')

plt.savefig('../out/HFT_lena')
plt.show()