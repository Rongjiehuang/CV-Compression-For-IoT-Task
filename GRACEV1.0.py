import cv2
import numpy as np
from function.GetWeights import getweights
from function.RGB2YUV import sourceRGB2YUV,sourceYUV2RGB
import os
import cv2
from PIL import Image

DirList = [
    'Pic/city'
]
# 块大小
Block_Size = 8
# 保存上一个块的DC系数
DC_y = 0
DC_cr = 0
DC_cb = 0
# zig zag方向
right = 0
down = 1
right_up = 2
left_down = 3
# 亮度量化表
Luminance_Quantization_Table = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]
# 色度量化表
Chrominance_Quantization_Table = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]


def quantize(block: np.ndarray, flag: int = 0):
    res = np.zeros((Block_Size, Block_Size), np.float32)
    if flag:
        # 色度量化
        for row in range(Block_Size):
            for col in range(Block_Size):
                res[row][col] = int(block[row][col] / Chrominance_Quantization_Table[row][col] + 0.5)
    else:
        # 亮度量化
        for row in range(Block_Size):
            for col in range(Block_Size):
                res[row][col] = int(block[row][col] / Luminance_Quantization_Table[row][col] + 0.5)
    return res


def inverse_quantize(block: np.ndarray, flag: int = 0):
    res = np.zeros((Block_Size, Block_Size), np.float32)
    if flag:
        # 色度量化
        for row in range(Block_Size):
            for col in range(Block_Size):
                res[row][col] = int(block[row][col] * Chrominance_Quantization_Table[row][col] + 0.5)
    else:
        # 亮度量化
        for row in range(Block_Size):
            for col in range(Block_Size):
                res[row][col] = int(block[row][col] * Luminance_Quantization_Table[row][col] + 0.5)
    return res


def encode(block: np.ndarray, flag: int):
    """
    对块进行DC差分编码和AC行程编码
    :param block:
    :return: 两个编码结果
    """
    global DC_y, DC_cr, DC_cb
    res_code = []
    # DC差分编码
    if flag == 0:
        res_code.append(block[0][0] - DC_y)
        DC_y = block[0][0]
    elif flag == 1:
        res_code.append(block[0][0] - DC_cr)
        DC_cr = block[0][0]
    elif flag == 2:
        res_code.append(block[0][0] - DC_cb)
        DC_cb = block[0][0]
    # AC行程编码
    zero_count = 0
    # zig zag遍历
    row = 0
    col = 1
    act = left_down
    while row != Block_Size and col != Block_Size:
        # 对当前数字进行处理
        if block[row][col]:
            res_code.append((zero_count, block[row][col]))
            zero_count = 0
        else:
            zero_count += 1
            if zero_count == 16:
                res_code.append((15, 0))
                zero_count = 0
            else:
                if row == Block_Size - 1 and col == Block_Size - 1:
                    res_code.append((zero_count - 1, 0))
        # 迭代
        if act == right:
            col += 1
            if row == 0:
                act = left_down
            else:
                act = right_up
        elif act == down:
            row += 1
            if col == 0:
                act = right_up
            else:
                act = left_down
        elif act == right_up:
            row -= 1
            col += 1
            if row == 0 and col < Block_Size - 1:
                act = right
            elif col == Block_Size - 1:
                act = down
            else:
                act = right_up
        elif act == left_down:
            row += 1
            col -= 1
            if row == Block_Size - 1:
                act = right
            elif col == 0 and row < Block_Size - 1:
                act = down
            else:
                act = left_down

    return res_code


def decode(code: list, flag):
    """
    对输入到code进行decode，[dc, (ac), (ac)....]
    :param code:
    :return:
    """
    global DC_y, DC_cr, DC_cb
    block = np.zeros((Block_Size, Block_Size))
    # DC系数复原
    if flag == 0:
        block[0][0] = np.int8(code[0] + DC_y)
        DC_y = block[0][0]
    elif flag == 1:
        block[0][0] = np.int8(code[0] + DC_cr)
        DC_cr = block[0][0]
    elif flag == 2:
        block[0][0] = np.int8(code[0] + DC_cb)
        DC_cb = block[0][0]
    # AC
    # 先从ac编码中得到具体值
    ac = []
    for i in range(1, len(code)):
        for j in range(code[i][0]):
            ac.append(0)
        ac.append(code[i][1])
    # zig zag遍历还原
    row = 0
    col = 1
    act = left_down
    count = 0
    while row != Block_Size and col != Block_Size:
        block[row][col] = np.int8(ac[count])
        count += 1
        # 迭代
        if act == right:
            col += 1
            if row == 0:
                act = left_down
            else:
                act = right_up
        elif act == down:
            row += 1
            if col == 0:
                act = right_up
            else:
                act = left_down
        elif act == right_up:
            row -= 1
            col += 1
            if row == 0 and col < Block_Size - 1:
                act = right
            elif col == Block_Size - 1:
                act = down
            else:
                act = right_up
        elif act == left_down:
            row += 1
            col -= 1
            if row == Block_Size - 1:
                act = right
            elif col == 0 and row < Block_Size - 1:
                act = down
            else:
                act = left_down
    return block


def encode_img(img: np.ndarray):
    # 将DC数据初始化
    global DC_y, DC_cb, DC_cr
    DC_y = 0
    DC_cr = 0
    DC_cb = 0
    # 保存code
    res_code = {'Y': [], 'Cr': [], 'Cb': [], 'shape': img.shape}
    # 转换颜色空间YCbCr
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # 图像分块及dct处理
    # 空图像块
    block_y: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    block_cr: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    block_cb: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    # 遍历做dct
    cols, rows = img.shape[0:2]
    for row in range(0, rows, Block_Size):
        for col in range(0, cols, Block_Size):
            # 取出大小为Block_Size的块，并做0偏置转化
            for i in range(Block_Size):
                for j in range(Block_Size):
                    block_y[i][j] = img[row + i][col + j][0] - 128
                    block_cr[i][j] = img[row + i][col + j][1] - 128
                    block_cb[i][j] = img[row + i][col + j][2] - 128

            # 对块进行dct处理
            block_y = cv2.dct(block_y)
            block_cr = cv2.dct(block_cr)
            block_cb = cv2.dct(block_cb)
            # 量化
            block_y = quantize(block_y, 0)
            block_cr = quantize(block_cr, 1)
            block_cb = quantize(block_cb, 1)
            # 编码
            res_code['Y'].append(encode(block_y, 0))
            res_code['Cr'].append(encode(block_cr, 1))
            res_code['Cb'].append(encode(block_cb, 2))
            # 实验不要求霍夫曼编码，即不进行熵编码
    # print(res_code['Y'])
    return res_code


def encode_img2(img: np.ndarray, WR, WG, WB):
    # 将DC数据初始化
    global DC_y, DC_cb, DC_cr
    DC_y = 0
    DC_cr = 0
    DC_cb = 0
    # 保存code
    res_code = {'Y': [], 'Cr': [], 'Cb': [], 'shape': img.shape}
    # 转换颜色空间YCbCr
    img: np.ndarray = sourceRGB2YUV(img,WR,WG,WB)

    # 图像分块及dct处理
    # 空图像块
    block_y: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    block_cr: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    block_cb: np.ndarray = np.zeros((Block_Size, Block_Size), np.float32)
    # 遍历做dct
    cols, rows = img.shape[0:2]
    for row in range(0, rows, Block_Size):
        for col in range(0, cols, Block_Size):
            # 取出大小为Block_Size的块，并做0偏置转化
            for i in range(Block_Size):
                for j in range(Block_Size):
                    block_y[i][j] = img[row + i][col + j][0] - 128
                    block_cr[i][j] = img[row + i][col + j][1] - 128
                    block_cb[i][j] = img[row + i][col + j][2] - 128

            # 对块进行dct处理
            block_y = cv2.dct(block_y)
            block_cr = cv2.dct(block_cr)
            block_cb = cv2.dct(block_cb)
            # 量化
            block_y = quantize(block_y, 0)
            block_cr = quantize(block_cr, 1)
            block_cb = quantize(block_cb, 1)
            # 编码
            res_code['Y'].append(encode(block_y, 0))
            res_code['Cr'].append(encode(block_cr, 1))
            res_code['Cb'].append(encode(block_cb, 2))
            # 实验不要求霍夫曼编码，即不进行熵编码
    # print(res_code['Y'])
    return res_code


def decode_img(code: dict):
    """
    根据code进行decode得到图片
    :param code:
    :return: img
    """
    # 将DC数据初始化
    global DC_y, DC_cb, DC_cr
    DC_y = 0
    DC_cr = 0
    DC_cb = 0
    # 生成一张新图片，经辛辛苦苦踩坑，必须设置为np.uint8
    img = np.zeros(code['shape'], dtype=np.uint8)

    # 每一块进行复原
    cols, rows = img.shape[0:2]
    for row in range(0, rows, Block_Size):
        for col in range(0, cols, Block_Size):
            # 取出每一块
            block_y = code['Y'][int((row / Block_Size) * (cols / Block_Size) + col / Block_Size)]
            block_cr = code['Cr'][int((row / Block_Size) * (cols / Block_Size) + col / Block_Size)]
            block_cb = code['Cb'][int((row / Block_Size) * (cols / Block_Size) + col / Block_Size)]
            # 对每一块进行解码
            block_y = decode(block_y, 0)
            block_cr = decode(block_cr, 1)
            block_cb = decode(block_cb, 2)
            # 对每一块进行反量化
            block_y = inverse_quantize(block_y, 0)
            block_cr = inverse_quantize(block_cr, 1)
            block_cb = inverse_quantize(block_cb, 1)
            # 对每一块进行逆DCT
            block_y = cv2.idct(block_y)
            block_cr = cv2.idct(block_cr)
            block_cb = cv2.idct(block_cb)
            # 对每一块进行复原，且取消0偏置转换
            for i in range(Block_Size):
                for j in range(Block_Size):
                    img[row + i][col + j][0] = block_y[i][j] + 128
                    img[row + i][col + j][1] = block_cr[i][j] + 128
                    img[row + i][col + j][2] = block_cb[i][j] + 128

    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

def GRACEV1(img,imgname):
    # img1 = img[:,0:1024,:]
    # img2 = img[:,1024:2048,:]
    img1 = img[:,0:128,:]
    img2 = img[:,128:256,:]

    WR, WG, WB = getweights(imgname)
    code1 = encode_img2(img1, WR, WG, WB)
    img1_decode = decode_img(code1)
    code2 = encode_img2(img2, WR, WG, WB)
    img2_decode = decode_img(code2)

    img_result = np.concatenate([img1_decode, img2_decode], axis=1)
    return img_result

if __name__ == '__main__':
    for path in DirList:
        for filename in os.listdir(path):
            fullName = os.path.join(path, filename)
            if fullName.endswith('.png'):
                mainName, ext = os.path.splitext(fullName)
                # print(fullName, mainName + '.jpeg')
                img = cv2.imread(fullName)
                img = cv2.resize(img,(256,128))
                GRACEV1(img,fullName)
                cv2.imwrite(mainName + '.jpeg', img)
                # im.save(mainName + '.jpeg', quality=80)
                # os.remove(fullName)

