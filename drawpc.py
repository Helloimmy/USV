# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/7/6 11:31
# @File    :点云绘制.py
# @Software:PyCharm
"""
@remarks :绘制三色点云，画出分割面和栅格，用于论文的图示
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 读取数据
    data = np.loadtxt("data_8.txt")
    r = data.shape[0]
    x_1, x_2, x_3 = np.zeros(0), np.zeros(0), np.zeros(0)
    y_1, y_2, y_3 = np.zeros(0), np.zeros(0), np.zeros(0)
    z_1, z_2, z_3 = np.zeros(0), np.zeros(0), np.zeros(0)
    for i in range(r):
        if -1 < data[i][0] <= -0.35:
            x_1 = np.append(x_1, data[i][0])
            y_1 = np.append(y_1, data[i][1])
            z_1 = np.append(z_1, data[i][2])
        elif -0.35 < data[i][0] <= 0.35:
            x_2 = np.append(x_2, data[i][0])
            y_2 = np.append(y_2, data[i][1])
            z_2 = np.append(z_2, data[i][2])
        elif 0.35 < data[i][0] <= 1:
            x_3 = np.append(x_3, data[i][0])
            y_3 = np.append(y_3, data[i][1])
            z_3 = np.append(z_3, data[i][2])
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_1, y_1, z_1, c='#14517C', marker='.', cmap='spectral', alpha=1)
    ax.scatter(x_2, y_2, z_2, c='#F3D266', marker='.', cmap='spectral', alpha=1)
    ax.scatter(x_3, y_3, z_3, c='#C497B2', marker='.', cmap='spectral', alpha=1)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.25, 1, 0.25, 0.5]))

    # 画分隔面
    # xx = np.linspace(-10.5, 10.5, 22)
    yy = np.linspace(-4, 4, 41)
    zz = np.linspace(0, 4, 21)
    Y, Z = np.meshgrid(yy, zz)
    # Z, M = np.meshgrid(zz, yy)
    for m_x in range(3):
        ax.plot_surface(X=Y*0+0.7*m_x-1.05,
                        Y=Y,
                        Z=Z,
                        color='#14517C',
                        alpha=0.1
                        )

    # 画网格
    x1 = []
    x2 = []
    x3 = []
    for x_idx in range(40):
        x1.append(-1.05)
        x2.append(-0.35)
        x3.append(0.35)
    y = [-40, -35, -30, -25, 25, 30, 35, 40, -40, -35, -30, -25, 25, 30, 35, 40,
         -40, -40, -40, -40, -40, -40, 40, 40, 40, 40, 40, 40, -12.5, 0, 12.5, -12.5,
         0, 12.5, -40, -40, -40, 40, 40, 40]
    z = [40, 40, 40, 40, 40, 40, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0,
         37, 34, 31, 9, 6, 3, 37, 34, 31, 9, 6, 3, 40, 40, 40, 0,
         0, 0, 25.5, 20, 14.5, 25.5, 20, 14.5]
    for yz_idx in range(40):
        y[yz_idx] /= 10
        z[yz_idx] /= 10

    A_1, B_1, C_1, D_1, E_1, F_1, G_1, H_1, A_2, B_2, C_2, D_2, E_2, F_2, G_2,\
    H_2, A_3, A_4, A_5, A_6, A_7, A_8, H_3, H_4, H_5, H_6, H_7, H_8, I_1, J_1,\
    K_1, I_2, J_2, K_2, I_3, J_3, K_3, I_4, J_4, K_4 = zip(x1, y, z)
    lines_1 = zip(A_1, H_1, H_2, A_2, A_1)
    lines_2 = zip(A_3, H_3, H_4, A_4, A_5, H_5, H_6, A_6, A_7, H_7, H_8, A_8)
    lines_3 = zip(B_1, B_2, C_2, C_1, D_1, D_2, E_2, E_1, F_1, F_2, G_2, G_1)
    lines_4 = zip(I_1, I_2, J_2, J_1, K_1, K_2)
    lines_5 = zip(I_3, I_4, J_4, J_3, K_3, K_4)
    ax.plot3D(*lines_1, zdir='z', c='gray')
    ax.plot3D(*lines_2, zdir='z', c='gray')
    ax.plot3D(*lines_3, zdir='z', c='gray')
    ax.plot3D(*lines_4, ls=':', zdir='z', c='gray')
    ax.plot3D(*lines_5, ls=':', zdir='z', c='gray')

    A_1, B_1, C_1, D_1, E_1, F_1, G_1, H_1, A_2, B_2, C_2, D_2, E_2, F_2, G_2, \
    H_2, A_3, A_4, A_5, A_6, A_7, A_8, H_3, H_4, H_5, H_6, H_7, H_8, I_1, J_1, \
    K_1, I_2, J_2, K_2, I_3, J_3, K_3, I_4, J_4, K_4 = zip(x2, y, z)
    lines_1 = zip(A_1, H_1, H_2, A_2, A_1)
    lines_2 = zip(A_3, H_3, H_4, A_4, A_5, H_5, H_6, A_6, A_7, H_7, H_8, A_8)
    lines_3 = zip(B_1, B_2, C_2, C_1, D_1, D_2, E_2, E_1, F_1, F_2, G_2, G_1)
    lines_4 = zip(I_1, I_2, J_2, J_1, K_1, K_2)
    lines_5 = zip(I_3, I_4, J_4, J_3, K_3, K_4)
    ax.plot3D(*lines_1, zdir='z', c='gray')
    ax.plot3D(*lines_2, zdir='z', c='gray')
    ax.plot3D(*lines_3, zdir='z', c='gray')
    ax.plot3D(*lines_4, ls=':', zdir='z', c='gray')
    ax.plot3D(*lines_5, ls=':', zdir='z', c='gray')

    A_1, B_1, C_1, D_1, E_1, F_1, G_1, H_1, A_2, B_2, C_2, D_2, E_2, F_2, G_2, \
    H_2, A_3, A_4, A_5, A_6, A_7, A_8, H_3, H_4, H_5, H_6, H_7, H_8, I_1, J_1, \
    K_1, I_2, J_2, K_2, I_3, J_3, K_3, I_4, J_4, K_4 = zip(x3, y, z)
    lines_1 = zip(A_1, H_1, H_2, A_2, A_1)
    lines_2 = zip(A_3, H_3, H_4, A_4, A_5, H_5, H_6, A_6, A_7, H_7, H_8, A_8)
    lines_3 = zip(B_1, B_2, C_2, C_1, D_1, D_2, E_2, E_1, F_1, F_2, G_2, G_1)
    lines_4 = zip(I_1, I_2, J_2, J_1, K_1, K_2)
    lines_5 = zip(I_3, I_4, J_4, J_3, K_3, K_4)
    ax.plot3D(*lines_1, zdir='z', c='gray')
    ax.plot3D(*lines_2, zdir='z', c='gray')
    ax.plot3D(*lines_3, zdir='z', c='gray')
    ax.plot3D(*lines_4, ls=':', zdir='z', c='gray')
    ax.plot3D(*lines_5, ls=':', zdir='z', c='gray')

    ho_1 = [-0.35, -3, 4]
    ho_2 = [-0.35, -2.5, 4]
    ho_3 = [-0.35, -3, 3.7]
    ho_4 = [-0.35, -2.5, 3.7]
    ho_5 = [0.35, -3, 4]
    ho_6 = [0.35, -2.5, 4]
    ho_7 = [0.35, -3, 3.7]
    ho_8 = [0.35, -2.5, 3.7]
    lines_6 = zip(ho_1, ho_5, ho_6, ho_8, ho_7, ho_3, ho_1, ho_2, ho_6, ho_5, ho_7)
    lines_7 = zip(ho_2, ho_4, ho_3, ho_4, ho_8)
    ax.plot3D(*lines_6, zdir='z', c='#D8383A', zorder=1000)
    ax.plot3D(*lines_7, ls=':', zdir='z', c='#D8383A', zorder=1000)

    ax.set(xlabel='X',
           ylabel='Y',
           zlabel='Z',
           xlim=(-1.05, 1.05),
           ylim=(-4, 4),
           zlim=(0, 4),
           # xticks=np.arange(0, 10, 2),
           # yticks=np.arange(0, 10, 1),
           # zticks=np.arange(0, 10, 1)
           )

    # 调整视角
    ax.view_init(elev=10,  # 仰角
                 azim=-90  # 方位角
                 )
    ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.w_xaxis.line.set_color('dimgrey')
    ax.w_yaxis.line.set_color('dimgrey')
    ax.w_zaxis.line.set_color('dimgrey')
    # plt.axis('off')
    # plt.savefig("cm.jpeg", dpi=2000)
    plt.show()


if __name__ == "__main__":
    main()
