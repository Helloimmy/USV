# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/8/27 18:50
# @File    :点云栅格化.py
# @Software:PyCharm
"""
@remarks :
"""
import numpy as np
import pandas as pd
import os
x_sum, y_sum, z_sum = 0, 0, 0

Data = np.loadtxt("data_8.txt")
num_array = np.zeros((3, 20, 10))
for j in range(len(Data)):
    x, y, z = Data[j][0], Data[j][1], Data[j][2]
    x_idx, y_idx, z_idx = int(((x*10)+10.5)//7), int(((y*10)+40)//4), int((z*10)//4)
    # print(x_idx, y_idx, z_idx)
    num_array[x_idx][y_idx][z_idx] += 1
# print(num_arra y.shape)
x_sum += np.sum(num_array[0])
y_sum += np.sum(num_array[1])
z_sum += np.sum(num_array[2])
print(x_sum, y_sum, z_sum)
# with open('re.txt', 'w') as outfile:
#     for slice_2d in num_array:
#         np.savetxt(outfile, slice_2d, fmt='%d', delimiter=' ')
# # c = np.loadtxt('/home/wln1/workspace/test6/b.txt', delimiter=' ').reshape((98, 78, 146))
sum_array = np.zeros((20, 10))

print(num_array[0].shape[0])
for num_s in range(20):
    for num_h in range(10):
        sum_array[num_s][num_h] = num_array[0][num_s][num_h] + num_array[1][num_s][num_h] + num_array[2][num_s][num_h]

sum_array = sum_array.T
data = pd.DataFrame(sum_array)

writer = pd.ExcelWriter('B.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()