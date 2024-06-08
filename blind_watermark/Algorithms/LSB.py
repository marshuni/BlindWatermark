import sys
import random
import numpy as np
from PIL import Image

class LSBSteganoser:
    def SetBit(self,var, num, value):
        mask = 1 << num
        var &= ~mask
        if value:
            var |= mask
        return var
    def Embed(self,src_img,payload_img):
        width,height = src_img.size
        payload = np.array(payload_img).flatten().tolist()

        # 发牌式往三个通道嵌入，所以载荷长度必须为3的倍数
        while(len(payload)%3):
            payload.append(0)

        steg_img = Image.new('RGB',(width,height))

        index = 0
        for h in range(height):
            for w in range(width):
                (r,g,b) = src_img.getpixel((w,h))
                if index < len(payload):
                    r = self.SetBit(r,0,payload[index])
                    g = self.SetBit(g,0,payload[index+1])
                    b = self.SetBit(b,0,payload[index+2])
                else:
                    r = self.SetBit(r,0,random.randint(0,1))
                    g = self.SetBit(g,0,random.randint(0,1))
                    b = self.SetBit(b,0,random.randint(0,1))
                steg_img.putpixel((w,h),(r,g,b))
                index += 3
        return steg_img
    
    def Extract(self,src_img):
        width,height = src_img.size
        data = []
        for h in range(height):
            for w in range(width):
                (r, g, b) = src_img.getpixel((w, h))
                data.append(r & 1)
                data.append(g & 1)
                data.append(b & 1)

        watermark_width = int(width*1.7)
        watermark_height = int(height*1.7)
        watermark = Image.new('1',(watermark_width,watermark_height))
        pixels = watermark.load()
        i = 0
        for y in range(watermark_height):
            for x in range(watermark_width):
                pixels[x,y] = data[i]
                i += 1
        return watermark
