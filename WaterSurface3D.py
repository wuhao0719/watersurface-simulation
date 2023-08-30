from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
import pandas as pd
import torch
from celluloid import Camera
import ffmpeg
import PIL


def H13(v):
    U = 0.0214*v**2
    return U


def SPhi(w, theta):
    C = 3.05
    U = 5
    g = 9.8
    S = C*(math.pi/(4*w**6)) * np.exp(-2 * g ** 2 / (U ** 2 * w ** 2))
    Phi = 8 / (3 * math.pi) * np.cos(theta)**4
    SPhi = S*Phi
    return SPhi

def SPhi2(w, theta, v):
    H = H13(v)
    S = (0.78/w**5) * np.exp(-3.11/(w**4*H**2))
    Phi = 2 / (math.pi) * (np.cos(theta)**2)
    SPhi = S*Phi
    return SPhi


def S(w, v):
    H = 0.0214*v**2
    S = 0.78/w**5 * np.exp(-3.11/(w**4*H**2))
    return S


def Alphadj(N):
    alphadj = np.arange(-math.pi/2, math.pi/2, math.pi/N)+math.pi/(2*N)
    return alphadj


def Fre(M, v):
    g = 9.8
    Index = np.arange(M+1)+1
    wi = (3.11/(H13(v)**2 * np.log((M+2)/Index)))**(1/4)
    #wdi = np.array((wi[1:M + 1] + wi[0:M]) / 2).reshape(1, M)
    #dw = (wi[1:M + 1] - wi[0:M]).reshape(1, M)
    #ki = wdi * wdi / g
    wdi = []
    dw = []
    for i in np.arange(M):
        wdi.append((wi[i]+wi[i+1])/2)
        dw.append(wi[i+1]-wi[i])

    wdi = np.array(wdi).reshape(1, M)
    dw = np.array(dw).reshape(1, M)
    ki = wdi*wdi/g

    return wdi, ki, dw


def Randombeta(M, N):
    beta = np.random.uniform(0,1,(M,N))
    return beta


def Constantbeta(M, N):
    beta = np.zeros((M, N))
    return beta

def Watersurface(M, N, v, t, x, y, beta):
    eit = 0
    dalpha = math.pi / N
    wdi, ki, dw = Fre(M, v)
    alphadj = Alphadj(N)
    wdiX, alphadjY = np.meshgrid(wdi, alphadj)
    ksi = (np.multiply(np.sqrt(2 * SPhi2(wdiX, alphadjY, v)).T, np.sqrt(np.tile(dw.T, (1, dw.size))))
           * np.sqrt(dalpha))
    for i in np.arange(M):
        for j in np.arange(N):
            #dalpha = math.pi/N
            #ksi = np.sqrt(2*SPhi2(wdi[0, i], alphadj[j], v)*dw[0, i]*dalpha)
            eit = eit+ksi[i, j]*np.cos(ki[0, i]*(x*np.cos(alphadj[j])+y*np.sin(alphadj[j]))-wdi[0, i]*t+beta[i, j])

    return eit


def Watersurfacegradient(M, N, v, t, x, y, beta):
    eitgradientx = 0
    eitgradienty = 0
    dalpha = math.pi / N
    wdi, ki, dw = Fre(M, v)
    alphadj = Alphadj(N)
    wdiX, alphadjY = np.meshgrid(wdi, alphadj)
    ksi = (np.multiply(np.sqrt(2 * SPhi2(wdiX, alphadjY, v)).T, np.sqrt(np.tile(dw.T, (1, dw.size))))
           * np.sqrt(dalpha))
    for i in np.arange(M):
        for j in np.arange(N):
            tempeitgradient = ksi[i, j]*np.sin(ki[0, i]*(x*np.cos(alphadj[j])+y*np.sin(alphadj[j]))-wdi[0, i]*t+beta[i, j])
            eitgradientx = eitgradientx-tempeitgradient*ki[0, i]*np.cos(alphadj[j])
            eitgradienty = eitgradienty-tempeitgradient*ki[0, i]*np.sin(alphadj[j])

    gradient = np.array([[eitgradientx, eitgradienty, 1]])
    return gradient


'''
writer = pd.ExcelWriter('beta.xlsx', engine='xlsxwriter')
DataFRame_data = pd.DataFrame(Randombeta(30, 30))
DataFRame_data.to_excel(writer, index=False)
writer.close()
'''


if __name__ == '__main__':
    T0 = 0
    Tend = 5
    Tsize = 50
    T = np.linspace(T0, Tend, Tsize)
    #准备x,y数据
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    #生成x、y网格化数据
    X, Y = np.meshgrid(x, y)
    #准备z值
    M=30
    N=30
    v=3

    df = pd.read_excel('beta.xlsx',  sheet_name=None)
    beta = np.array(df.get('Sheet1'))
    #beta = Constantbeta(M, N)



    '''
    t=0
    Z = Watersurface(M, N, v, t, X, Y, beta)
    #绘制图像
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #调用绘制线框图的函数plot_wireframe()
    ax.plot_wireframe(X, Y, Z)
    ax.set_title('wireframe')
    plt.show(block=True)
    '''

    Z = np.zeros((Tsize, x.size, y.size))
    eitgradient = np.zeros((2, Tsize, x.size, y.size))
    for Index in range(Tsize):
        t = T[Index]
        print(t)
        Z[Index, :, :] = Watersurface(M, N, v, t, X, Y, beta)
        #eitgradient[Index, :, :] = Watersurfacegradient(M, N, v, t, X, Y, beta)[0]

    azim = -60
    elev = 30

    plt.ion()
    fig = plt.figure()
    camera = Camera(fig)
    ax = fig.add_axes(Axes3D(fig))
    ax.view_init(elev, azim)  # 设定角度
    for i in range(Tsize):
        ax.plot_surface(X, Y, Z[i, :, :], cmap='rainbow')
        #plt.pause(1)  # 暂停一段时间，不然画的太快会卡住显示不出来
        camera.snap()
    animation = camera.animate()
    animation.save('celluloid_minimal2.gif',
                   writer='pillow', fps=10)


'''
    # 创建绘制实时损失的动态窗口
    plt.ion()
    for i in range(Tsize):
        plt.clf()  # 清除之前画的图
        fig = plt.gcf()  # 获取当前图
        #ax = fig.gca(projection='3d')  # 获取当前轴
        ax = fig.add_axes(Axes3D(fig))
        ax.view_init(elev, azim)  # 设定角度

        ax.plot_surface(X, Y, Z[i, :, :], cmap='rainbow')
        plt.pause(0.01)  # 暂停一段时间，不然画的太快会卡住显示不出来

        elev, azim = ax.elev, ax.azim  # 将当前绘制的三维图角度记录下来，用于下一次绘制（放在ioff()函数前后都可以，但放在其他地方就不行）
        # elev, azim = ax.elev, ax.azim - 1 # 可自动旋转角度，不需要人去拖动

        plt.ioff()  # 关闭画图窗口Z

    # 加这个的目的是绘制完后不让窗口关闭
    plt.show()
'''


