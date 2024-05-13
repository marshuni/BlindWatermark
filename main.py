import sys
import os

from PIL import Image
from converter import Converter
from Algorithms.LSB import LSBSteganoser
from Algorithms.DFT import DFTSteganoser
from Algorithms.DWT import DWTSteganoser


def hide(img_file, payload_file, mode):
    # 读取载体文件 确认输出目录
    carrier_img = Image.open(img_file).convert("RGB")
    img_ext = os.path.splitext(img_file)[1]
    width,height = carrier_img.size
    steg_path = os.path.splitext(img_file)[0] + '-steg-' + mode + img_ext

    # 读取水印文字or包含水印文本文件
    # print(payload_file)
    payload_text = ""
    payload_ext = os.path.splitext(payload_file)[1][1:]
    if os.path.exists(payload_file) and payload_ext == 'txt':
        with open(payload_file,"r",encoding="utf-8") as payload:
            payload_text = payload.read()
    elif payload_file[0]=='*':
        payload_text = payload_file.strip('*')
    else:
        print("[-] Unsupported payload file format. Try to input path of a .txt file or a string starting with \'*\'.")
        sys.exit()

    # 转换为载体图像 不同嵌入格式可嵌入的图像尺寸有差异。
    if mode=='LSB':
        pwidth = int(width*1.7)
        pheight = int(height*1.7)
    elif mode=='DFT':
        pwidth = width
        pheight = int(height*0.45)
    elif mode=='DWT':
        pwidth = int(width/8)
        pheight = int(height/8)
    payload_img = Converter().Text2Pic(payload_text,pwidth,pheight)
    
    # print("[*] Please preview the payload image and then close tha window.")
    # payload_img.show()
    
    # 转换并保存
    if mode=='LSB':
        steg_img = LSBSteganoser().Embed(carrier_img,payload_img)
    elif mode=='DFT':
        steg_img = DFTSteganoser().Embed(carrier_img,payload_img)
    elif mode=='DWT':
        steg_img = DWTSteganoser().Embed(carrier_img,payload_img)
    steg_img.save(steg_path)
    print ("[+] Embedded successfully!")

def extract(steg_file,output_file,mode):
    
    steg_img = Image.open(steg_file).convert("RGB")

    if mode=='LSB':
        payload_img = LSBSteganoser().Extract(steg_img)
    elif mode=='DFT':
        payload_img = DFTSteganoser().Extract(steg_img)
    elif mode=='DWT':
        payload_img = DWTSteganoser().Extract(steg_img)
    payload_img.save(output_file)
    print ("[+] Extracted successfully!")

def check():
    arg = 5
    if len(sys.argv)<arg:
        print("[-] Too few arguments provided. Please check your input.")
        sys.exit()
    elif len(sys.argv)>arg:
        print("[-] Too many arguments provided. Please check your input.")
        sys.exit()

    if sys.argv[4] not in ['LSB','DFT','DWT']:
        print("[-] Invalid mode.")
        sys.exit()


    img_ext = os.path.splitext(sys.argv[2])[1][1:]
    if img_ext not in ['jpeg','jpg','png','bmp']:
        print("[-] Unsupported carrier image file.")
        sys.exit()
   
def usage(ScriptName):
    print("隐写工具：可使用该脚本将文本或图片(灰度)藏入载体图像的最低位。脚本用法详见中文文档。")
    print("-------------------")
    print("LSB steganogprahy. You can hide texts or small pics(gray-scale) within least significant bits of images.\n")

    print("Usage:")
    print("  %s hide <img_file> <payload_file> <mode>" % ScriptName)
    print("  %s extract <stego_file> <output_file> <mode>\n" % ScriptName)

    print("<mode>: LSB | DFT | DWT")
    print("<output_file>: A .png file.")

    sys.exit()

if __name__ == "__main__":
    if len(sys.argv)<2 or sys.argv[1]=="help":
        usage(sys.argv[0])
    elif sys.argv[1]=="hide":
        check()
        hide(sys.argv[2],sys.argv[3],sys.argv[4])
    elif sys.argv[1]=="extract":
        check()
        extract(sys.argv[2],sys.argv[3],sys.argv[4])
    else:
        print("Invalid Command. Use \"%s help\" for help." % sys.argv[0])
    