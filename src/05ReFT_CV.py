# -*- coding: UTF-8 -*-
"""
@Project : FT_Pro
@File    : 05ReFT_CV
@IDE     : PyCharm
@Author  : 李睡睡
@Date    : 2022/12/4 3:08
@Scripts : opencv 、 numpy 、 matplotlib
@Documentation:https://www.bilibili.com/video/BV1qt4y1u7N9?p=6&vd_source=fa4736bfe6ad0afc755decfbf72b082a
@Function: 傅里叶逆变换 的 cv 实现
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)

# 与04一致，这里先是正向的傅里叶变换
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)

# 移动低频位置
dftShift = np.fft.fftshift(dft)
ishift = np.fft.ifftshift(dftShift)

# 逆的傅里叶变换
iimg = cv2.idft(ishift)

# 计算幅度
iimg = cv2.magnitude(iimg[:,:,0],iimg[:,:,1])

# 显示原始图像
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original'),plt.axis('off')

# 显示滤波结果
plt.subplot(122),plt.imshow(iimg,cmap='gray')
plt.title('inverse'),plt.axis('off')

plt.show()