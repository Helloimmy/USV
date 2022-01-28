# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/9/14 10:48
# @File    :船图片提取RGB.py
# @Software:PyCharm
"""
@remarks :
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def produceImage(image, width, height):
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    return resized_image


path2 = "D:/HUST/prcharm_env_pytorch/test12/船.jpg"
data2 = Image.open(path2)  # 读取图片
data2 = produceImage(data2, 20, 10)

data2 = np.array(data2)
print(data2.shape)

plt.imshow(data2)
plt.show()

