# -*- coding: UTF-8 -*-
"""
@Project ：FT_Pro 
@File    ：FT_Demo.py
@IDE     ：PyCharm 
@Author  ：李睡睡
@Date    ：2022/12/2 15:48 
@Scripts :opencv 、 numpy 、 matplotlib
@Function:实现了 傅里叶变换 高低频显示
@Tips    :不要左顾右盼。慢慢积累，慢慢写吧。毕竟除了这样单调的努力，我什么也做不了。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image/lena.png',0)


# 进行傅里叶变换，f为复数形式
f = np.fft.fft2(img)

# 将低频转移到中心，fshift仍为复数
fshift = np.fft.fftshift(f)

# 使用abs取绝对值，再取log值，*20，值为可显示的频谱
result = 20*np.log(np.abs(fshift))

# 傅里叶变换前
# 121 表示创建显示窗口，
plt.subplot(121)
plt.imshow(img,cmap='gray')
# 添加标题
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(result,cmap='gray')
# 去掉坐标轴
plt.axis('off')

plt.savefig('../out/FT_lena')
plt.show()

