import sys
from PIL import Image

class DWTSteganoser:
    def Embed(self,src_img,payload_img):
        # src_img:载体图像; payload_img:水印图像;都为Image对象
        width,height = src_img.size

        
        # 返回叠加水印后的载体图像
        steg_img = Image.new('RGB',(width,height))

        print ("[+] Embedded successfully!")
        return steg_img
    
    def Extract(self,src_img):
        # src_img:载体图像;为Image对象
        width,height = src_img.size
        print ("[*] Input image size: %dx%d pixels." % (width, height))

        # 返回水印图像
        # 注意：算法限制，最多只能嵌入1/8倍原图尺寸的数据
        watermark_img = Image.new('RGB',(int(width/8),int(height/8)))
        
        print ("[+] Extracted successfully!")
        return watermark_img
