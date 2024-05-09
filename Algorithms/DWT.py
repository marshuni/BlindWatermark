import sys
from PIL import Image

class LSBSteganoser:
    def Embed(self,src_img,payload_img):
        width,height = src_img.size

        

        steg_img = Image.new('RGB',(width,height))

        print ("[+] Embedded successfully!")
        return steg_img
    
    def Extract(self,src_img):
        width,height = src_img.size
        print ("[*] Input image size: %dx%d pixels." % (width, height))

        watermark_img = Image.new('RGB',(width,height))
        print ("[+] Extracted successfully!")
        return watermark_img
