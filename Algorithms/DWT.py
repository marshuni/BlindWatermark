import sys
from PIL import Image
import numpy as np
import cv2
from numpy.linalg import svd
import copy
from cv2 import dct, idct
from pywt import dwt2, idwt2
import multiprocessing
import warnings

class CommonPool(object):
    def map(self, func, args):
        return list(map(func, args))

class AutoPool(object):
    def __init__(self):

        self.mode = 'common'
        self.processes = None
        self.pool = CommonPool()   

    def map(self, func, args):
        return self.pool.map(func, args)


class WaterMarkCore:
    def __init__(self, password_img=1):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大

        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = AutoPool()

        self.alpha = None  # 用于处理透明图

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        # 处理透明图
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        # dct->svd->打水印->逆svd->逆dct
        block, shuffler, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))
        
    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # 4维分块变回2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        block, shuffler = args
        # dct->svd->解水印
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm
    
    def extract_raw(self, img):
        # 每个分块提取 1 bit 信息
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来

        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=(wm_block_bit.shape[1]))
        # print("#")
        # print(wm_avg.shape)
        # print(wm_block_bit.shape)
        for i in range(0,wm_block_bit.shape[1]):
            wm_avg[i] = wm_block_bit[:, i].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()
        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_raw(img=img)
        print(wm_block_bit.shape)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(4, 4)):
        self.bwm_core = WaterMarkCore(password_img=password_img)

        self.password_wm = password_wm

        self.wm_bit = None
        self.wm_size = 0

    def read_img(self, img):

        # 从文件读入图片
        img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

        self.bwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm):
        wm = np.array(wm)
        # 将二值化图像转换为灰度图像
        wm = np.where(wm > 0, 255, 0).astype(np.uint8)
        wm = cv2.cvtColor(wm,cv2.IMREAD_GRAYSCALE)

        # 读入图片格式的水印，并转为一维 bit 格式，抛弃灰度级别
        self.wm_bit = wm.flatten() > 0

        self.wm_size = self.wm_bit.size

        # 水印加密:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None):
        
        embed_img = self.bwm_core.embed()

        cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(wm_avg.size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, embed_img=None, wm_shape=None):

        # embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
        embed_img = self.read_img(embed_img)
        self.wm_size = np.array(wm_shape).prod()
        
        wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)
        print(wm_avg.shape)
        # 解密：
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # 转化为指定格式：
        wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
        wm = 255*wm

        return wm

class DWTSteganoser:
    def Embed(self,src_img,payload_img):
        # src_img:载体图像; payload_img:水印图像;都为Image对象

        bwm = WaterMark(password_wm=1, password_img=1)
        
        # 读取原图
        bwm.read_img(src_img)

        # 读取水印
        bwm.read_wm(payload_img)

        # 打上盲水印
        steg_op = bwm.embed('embedded.png')
        
        steg_img = cv2.cvtColor(steg_op, cv2.COLOR_BGR2RGB)
        steg_img = np.array(steg_img,dtype=np.uint8)
        steg_img = Image.fromarray(steg_img)

        print ("[+] Embedded successfully!")
        return steg_img
    
    def Extract(self,src_img):
        # src_img:载体图像;为Image对象
        width,height = src_img.size
        print ("[*] Input image size: %dx%d pixels." % (width, height))
        
        # 返回水印图像
        # 注意：算法限制，最多只能嵌入1/8倍原图尺寸的数据
        watermark_op = Image.new('RGB',(int(width/8),int(height/8)))
        wm_shape =  cv2.cvtColor(np.asarray(watermark_op),cv2.IMREAD_GRAYSCALE).shape
        
        # %% 解水印
        bwm1 = WaterMark(password_wm=1, password_img=1)

        # 注意需要设定水印的长宽wm_shape
        watermark_op = bwm1.extract(src_img, wm_shape=wm_shape)
        steg_img = cv2.convertScaleAbs(watermark_op, alpha=(255/65535.0))
        steg_img = cv2.cvtColor(steg_img,cv2.COLOR_BGR2RGB)
        steg_img = np.array(steg_img,dtype=np.uint8)
        print(steg_img.shape)
        steg_img = Image.fromarray(steg_img)
        print ("[+] Extracted successfully!")
        return steg_img