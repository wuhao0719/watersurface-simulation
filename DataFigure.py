import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from celluloid import Camera
import ffmpeg
import PIL


Radius = 0.015 #m
L = 0.03 #m
T0 = 0
Tend = 1
Tsize = 10
scale = 1
T = np.linspace(T0, Tend, Tsize)
x = np.array([-L, 0, L])
y = np.array([-L, 0, L])
Locationx, Locationy = np.meshgrid(x, y)
RLocationx = Locationx.flatten()
RLocationy = Locationy.flatten()
circle1 = plt.Circle((RLocationx[0], RLocationy[0]), Radius)
circle2 = plt.Circle((RLocationx[1], RLocationy[1]), Radius)
circle3 = plt.Circle((RLocationx[2], RLocationy[2]), Radius)
circle4 = plt.Circle((RLocationx[3], RLocationy[3]), Radius)
circle5 = plt.Circle((RLocationx[4], RLocationy[4]), Radius)
circle6 = plt.Circle((RLocationx[5], RLocationy[5]), Radius)
circle7 = plt.Circle((RLocationx[6], RLocationy[6]), Radius)
circle8 = plt.Circle((RLocationx[7], RLocationy[7]), Radius)
circle9 = plt.Circle((RLocationx[8], RLocationy[8]), Radius)

def PlotCircle(a, b, r):
    t = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(t)
    y = b + r * np.sin(t)
    return np.array([x, y])

#df = pd.read_excel('Data104.xlsx', sheet_name=None)
df2 = pd.read_excel('DataRB813_010100105.xlsx', sheet_name=None)

'''
plt.ion()
fig = plt.figure()
ax = plt.gca()
camera = Camera(fig)
for i in range(Tsize):
    #plt.clf()  # 清除之前画的图
    #fig = plt.gcf()  # 获取当前图
    #ax = plt.gca()
    #ax = fig.gca(projection='3d')  # 获取当前轴
    #if np.array(df.get(str(i))).size != 0:
    #    plt.scatter(np.array(df.get(str(i)))[:, 0], np.array(df.get(str(i)))[:, 1], linewidth=2)
    if np.array(df2.get(str(i))).size != 0:
        plt.scatter(np.array(df2.get(str(i)))[:, 0], np.array(df2.get(str(i)))[:, 1], linewidth=2)
    plt.plot(PlotCircle(RLocationx[0], RLocationy[0], Radius)[0,:] , PlotCircle(RLocationx[0], RLocationy[0], Radius)[1,:], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[1], RLocationy[1], Radius)[0, :],
             PlotCircle(RLocationx[1], RLocationy[1], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[2], RLocationy[2], Radius)[0, :],
             PlotCircle(RLocationx[2], RLocationy[2], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[3], RLocationy[3], Radius)[0, :],
             PlotCircle(RLocationx[3], RLocationy[3], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[4], RLocationy[4], Radius)[0, :],
             PlotCircle(RLocationx[4], RLocationy[4], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[5], RLocationy[5], Radius)[0, :],
             PlotCircle(RLocationx[5], RLocationy[5], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[6], RLocationy[6], Radius)[0, :],
             PlotCircle(RLocationx[6], RLocationy[6], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[7], RLocationy[7], Radius)[0, :],
             PlotCircle(RLocationx[7], RLocationy[7], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[8], RLocationy[8], Radius)[0, :],
             PlotCircle(RLocationx[8], RLocationy[8], Radius)[1, :], linewidth=2, color='red')
    plt.xlim((-L - Radius)*scale, (L + Radius)*scale)
    plt.ylim((-L - Radius)*scale, (L + Radius)*scale)
    ax.set_xlabel(r'x (m)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(r'y (m)', fontsize=14, fontname='Times New Roman')
    plt.grid(True)
    plt.pause(0.1)  # 暂停一段时间，不然画的太快会卡住显示不出来
    camera.snap()
    #plt.ioff()  # 关闭画图窗口Z
# 加这个的目的是绘制完后不让窗口关闭
plt.show()
'''


fig = plt.figure()
camera = Camera(fig)
ax = plt.gca()
for i in range(Tsize):

    if np.array(df2.get(str(i))).size != 0:
        plt.scatter(np.array(df2.get(str(i)))[:, 0], np.array(df2.get(str(i)))[:, 1],
                    linewidth=2, color='b')
    plt.plot(PlotCircle(RLocationx[0], RLocationy[0], Radius)[0, :],
             PlotCircle(RLocationx[0], RLocationy[0], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[1], RLocationy[1], Radius)[0, :],
             PlotCircle(RLocationx[1], RLocationy[1], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[2], RLocationy[2], Radius)[0, :],
             PlotCircle(RLocationx[2], RLocationy[2], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[3], RLocationy[3], Radius)[0, :],
             PlotCircle(RLocationx[3], RLocationy[3], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[4], RLocationy[4], Radius)[0, :],
             PlotCircle(RLocationx[4], RLocationy[4], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[5], RLocationy[5], Radius)[0, :],
             PlotCircle(RLocationx[5], RLocationy[5], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[6], RLocationy[6], Radius)[0, :],
             PlotCircle(RLocationx[6], RLocationy[6], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[7], RLocationy[7], Radius)[0, :],
             PlotCircle(RLocationx[7], RLocationy[7], Radius)[1, :], linewidth=2, color='red')
    plt.plot(PlotCircle(RLocationx[8], RLocationy[8], Radius)[0, :],
             PlotCircle(RLocationx[8], RLocationy[8], Radius)[1, :], linewidth=2, color='red')
    plt.xlim((-L - Radius) * scale, (L + Radius) * scale)
    plt.ylim((-L - Radius) * scale, (L + Radius) * scale)
    ax.set_xlabel(r'x (m)', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(r'y (m)', fontsize=14, fontname='Times New Roman')
    plt.grid(True)
    camera.snap()

animation = camera.animate()
animation.save('celluloid_minimal3.gif',
               writer='pillow', fps=5)