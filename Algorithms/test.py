import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('avatar.png',0) #直接读为灰度图像
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)#将图像中的低频部分移动到图像的中心
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift))
print(s1)
plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')
plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')

plt.show()
