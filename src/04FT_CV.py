# -*- coding: UTF-8 -*-
"""
@Project : FT_Pro
@File    : 04FT_CV
@IDE     : PyCharm
@Author  : 李睡睡
@Date    : 2022/12/4 3:00
@Scripts :opencv 、 numpy 、 matplotlib
@Documentation:https://www.bilibili.com/video/BV1qt4y1u7N9?p=6&vd_source=fa4736bfe6ad0afc755decfbf72b082a
@Function: 傅里叶变换的 cv 实现
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)

# 傅里叶变换,由cv得到了 图像的 频谱
dft = cv2.dft(np.float32(img),flags= cv2.DFT_COMPLEX_OUTPUT)

dftShift = np.fft.fftshift(dft)

# 之前得到的是傅里叶的一个结果，但其为双通道，使用magnitude函数进行转换
result = 20*np.log(cv2.magnitude(dftShift[:,:,0],dftShift[:,:,1]))

# 显示原始图像
plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original'),plt.axis('off')

# 显示滤波结果
plt.subplot(122),plt.imshow(result,cmap='gray')
plt.title('result'),plt.axis('off')

plt.show()