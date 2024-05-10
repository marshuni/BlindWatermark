import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

debug=0
class DFTSteganoser:
    def Embed(self, src_img: Image.Image, payload: Image.Image):
        src_img = np.array(src_img.convert("RGB"))
        payload = np.array(payload.convert("1"))

        # 输出原始图像
        if debug:
            plt.subplot(231), plt.imshow(src_img), \
                plt.title('Image_Raw')
            plt.xticks([]), plt.yticks([])
        
        # 处理水印
        # 中心对称的水印有助于增强图像的隐蔽性
        watermark = np.zeros(src_img.shape)
        for i in range(payload.shape[0]):
            for j in range(payload.shape[1]):
                watermark[i][j] = ~payload[i][j]
                watermark[watermark.shape[0]-i-1][watermark.shape[1]-j-1] = ~payload[i][j]

        # 输出处理后的水印
        if debug:
            plt.subplot(232), plt.imshow(watermark), \
                plt.title('Watermark')
            plt.xticks([]), plt.yticks([])
        
        # 进行离散傅里叶变换并中心化 先建好各个数组
        f = np.zeros_like(src_img, dtype=np.complex128)
        fshift = np.zeros_like(src_img, dtype=np.complex128)
        fshift_wm = np.zeros_like(src_img, dtype=np.complex128)
        f_wm = np.zeros_like(src_img, dtype=np.complex128)
        
        for i in range(src_img.shape[2]):  # 遍历每个通道
            f[:,:,i] = np.fft.fft2(src_img[:,:,i])
            fshift[:,:,i] = np.fft.fftshift(f[:,:,i])
            # fshift_wm[:,:,i] = fshift[:,:,i]

            for y in range(src_img.shape[0]):
                for x in range(src_img.shape[1]):
                    # 将存在水印处置为-1，即
                    # 由于水印位于图像高频处，削去高频的隐蔽性比加强高频可能更好
                    # 此处-1仅为临时取值 取什么值更好需要经过试验
                    if watermark[y,x,0]:
                        fshift_wm[y,x,i] = -1 + fshift_wm[y,x,i].imag
                    else:
                        fshift_wm[y,x,i] = fshift[y,x,i]
            f_wm[:,:,i] = np.fft.fftshift(fshift_wm[:,:,i])

        # 频域图像预览
        if debug:
            freq = np.log(np.abs(fshift_wm))
            freq = np.array((freq-np.min(freq))*255 / (np.max(freq)-np.min(freq)),dtype=np.int8)
            s = Image.fromarray(freq,"RGB")
            plt.subplot(233), plt.imshow(s), \
                plt.title('Frequency Domain')
            plt.xticks([]), plt.yticks([])

        # 还原成空域的图像
        steg_img = np.zeros(src_img.shape)
        for i in range(src_img.shape[2]):  # 遍历每个通道
            steg_img[:,:,i] = np.real(np.fft.ifft2(f_wm[:,:,i]))
        
        # steg_img = np.array((steg_img-np.min(steg_img))*255 / (np.max(steg_img)-np.min(steg_img)),dtype=np.float32)
        steg_img = steg_img.astype(int)
        steg_img = Image.fromarray(np.uint8(steg_img))

        if debug:
            plt.subplot(234), plt.imshow(steg_img), \
                plt.title('Image_Watermark')
            plt.xticks([]), plt.yticks([])
        if debug:
            plt.show()
        return steg_img
    
    def Extract(self,src_img):
        src_img = np.array(src_img.convert("RGB"))

        f = np.zeros_like(src_img, dtype=np.complex128)
        fshift = np.zeros_like(src_img, dtype=np.complex128)
        
        for i in range(src_img.shape[2]):  # 遍历每个通道
            f[:,:,i] = np.fft.fft2(src_img[:,:,i])
            fshift[:,:,i] = np.fft.fftshift(f[:,:,i])

        freq = np.log(np.abs(fshift))
        freq = np.array((freq-np.min(freq))*255 / (np.max(freq)-np.min(freq)),dtype=np.int8)
        watermark = Image.fromarray(freq,"RGB")

        enhancer = ImageEnhance.Contrast(watermark)
        watermark = enhancer.enhance(1.5)
        # watermark = watermark.filter(ImageFilter.SHARPEN)
        # watermark.show()
        return watermark