import numpy as np
from PIL import Image

class DFTSteganoser:
    def embed(self, src_img: Image.Image, payload: Image.Image):
        width,height = src_img.size
        payload_width,payload_height = payload.size

        # 处理水印图像
        # if payload_height*2>height or payload_width*2>width:
        #     payload = payload.resize(width*0.45,height*0.45)
        #     print("水印图像尺寸过大，将进行缩放。")
        

        src_img = np.array(src_img.convert("RGB"))
        payload = np.array(payload.convert("1"))

        watermark = np.zeros(src_img.shape)

        f = np.fft.fft2(src_img,)
        # fshift = np.fft.fftshift(f)

        for i in range(src_img.shape[2]):  # 遍历每个通道
            f[:,:,i] = np.fft.fft2(src_img[:,:,i])
        tmp = np.log(np.abs(f))
        print(tmp)
        tmp = np.array((tmp-np.min(tmp))*255 / (np.max(tmp)-np.min(tmp)),dtype=np.int8)

        # import matplotlib.pyplot as plt
        # plt.imshow(tmp,'gray')
        # plt.show()

        print(tmp.shape)
        s = Image.fromarray(tmp,"RGB")
        s.show()
        # s.show()

        
        
        pass

src_img = Image.open("avatar.png")
payload = Image.open("payload.png")
d = DFTSteganoser()
d.embed(src_img,payload)
