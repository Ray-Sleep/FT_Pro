# -*- coding: UTF-8 -*-
"""
@Project : FT_Pro
@File    : 06LPF_Demo
@IDE     : PyCharm
@Author  : 李睡睡
@Date    : 2022/12/4 3:26
@Scripts :opencv 、 numpy 、 matplotlib
@Documentation:https://www.bilibili.com/video/BV1qt4y1u7N9?p=6&vd_source=fa4736bfe6ad0afc755decfbf72b082a
@Function: 低通滤波的 cv 实现
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)

dft = cv2.dft(np.float32(img),flags= cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

# 构造掩膜图像
rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 1

# 滤波过程
fShift = dftShift*mask

# 再进行逆向的 傅里叶变换
ishift = np.fft.ifftshift(fShift)
iimg = cv2.idft(ishift)
iimg = cv2.magnitude(iimg[:,:,0],iimg[:,:,1])

# 显示原始图像
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original'),plt.axis('off')

# 显示 低通滤波 结果
plt.subplot(122),plt.imshow(iimg,cmap='gray')
plt.title('inverse'),plt.axis('off')

plt.savefig('../out/LPF_lena')
plt.show()