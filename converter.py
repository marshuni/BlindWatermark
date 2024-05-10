import struct
import sys
from PIL import Image, ImageFont, ImageDraw

class Converter:
    @staticmethod
    def TextSplit(text:str, width):
        result = []
        current_width = 0
        current_line = ""

        for char in text:
            char_width = 0.5 if 0<=ord(char)<=127 else 1  # 全角字符宽度为2，半角字符宽度为1

            if current_width + char_width + 1 > width:
                result.append(current_line)
                current_line = ""
                current_width = 0

            current_line += char
            current_width += char_width

        result.append(current_line)

        return result
    
    @staticmethod
    def Text2Pic(raw,width,height) -> Image:
        sections = raw.split(sep='\n')
        text = ""
        margin = 10 if width>64 else 2

        # 计算合适的字体大小
        size = 14
        for siz in [8,9,10,12,14,16,20,22,24,26,28,36,48,72,144,180,288,300,360,480]:
            column = (width-margin*2)/siz
            line = len(raw)/column + len(sections)
            if line*(siz+4)<height:
                size = siz

        for section in sections:
            for line in Converter.TextSplit(section,int((width-margin*2)/size)):
                text += line+'\n'
        
        image = Image.new("1", (width, height),255)
        draw = ImageDraw.Draw(image)
        # Available Font：Simsun.ttc / YaHeiConsolas.ttf
        # Put your own font file into ./Fonts/ 
        font = ImageFont.truetype("./Fonts/YaHeiConsolas.ttf", size)

        draw.multiline_text((margin, margin/2), text, font=font, fill="#000000",spacing=4)
        return image

