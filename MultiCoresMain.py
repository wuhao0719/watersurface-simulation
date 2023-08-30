import numpy as np
import math
from sympy import *
import sympy as sp
import WaterSurface3D
import scipy
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rcParams
import pandas as pd
from multiprocessing.pool import Pool

def Photonsgenerater(PhotonsNumber):
    xp = np.zeros((PhotonsNumber, 1))
    yp = np.zeros((PhotonsNumber, 1))
    zp = -ZTx*np.ones((PhotonsNumber, 1)) #xp yp zp Position
    W = 1*np.ones((PhotonsNumber, 1))
    flag = np.zeros((PhotonsNumber, 1)) #-1 die     0 undewater 1 crosswater 2 air
    RFlag = -np.ones((PhotonsNumber, 1))
    # special light
    ZetaPhi = np.random.rand(PhotonsNumber, 1)
    ZetaTheta = np.random.rand(PhotonsNumber, 1)
    Phiini = 2*math.pi*ZetaPhi #0-2pi
    Thetaini = np.arccos(1-ZetaTheta*(1-np.cos(np.radians(0.5*PhiT)))) #0-2pi
    PhotonsDirection = np.concatenate((np.cos(Phiini)*np.sin(Thetaini), np.sin(Phiini)*np.sin(Thetaini), np.cos(Thetaini)), axis=1)

    PhotonsPosition = np.concatenate((xp, yp, zp), axis=1)
    PhotonsMatrix = np.concatenate((PhotonsPosition, PhotonsDirection, W, flag, RFlag), axis=1)
    return PhotonsMatrix


def ArrivalJudgement(PhotonsPosition, PhotonsDirection, RLocationx, RLocationy, RLocationz, Radius, PhiR):
    xp, yp, zp = PhotonsPosition
    nx, ny, nz = PhotonsDirection
    flag = -1
    t = (-zp+RLocationz[0])/nz
    ArrivalPosition = np.array([xp+nx*t, yp+ny*t, RLocationz[0]])
    flag2 = np.degrees(np.arctan(np.sqrt(nx ** 2 + ny ** 2) / nz)) < PhiR / 2
    for Index in np.arange(RLocationx.size):
        xRx = RLocationx[Index]
        yRx = RLocationy[Index]
        zRx = RLocationz[Index]
        flag1 = (xp+nx*(zRx-zp)/nz-xRx)**2+(yp+ny*(zRx-zp)/nz-yRx)**2 < Radius**2
        if flag1 == 1 and flag2 == 1:
            flag = Index
            break
    #flag1 = (xp+nx*t)**2+(yp+ny*t)**2 < 0.1**2
    #if flag1 == 1 and flag2 == 1:
    #    flag = 1

    return flag, ArrivalPosition


def func(x, PhotonsPosition = np.array([0,0,0]), PhotonsDirection= np.array([0,0,0]), M = 1, N = 1, v = 0, t = 0, beta = np.zeros((1,1))):
    PhotonsPositionx, PhotonsPositiony, PhotonsPositionz = PhotonsPosition
    PhotonsDirectionx, PhotonsDirectiony, PhotonsDirectionz = PhotonsDirection
    eit = WaterSurface3D.Watersurface(M, N, v, t, PhotonsPositionx + x * PhotonsDirectionx,
                                      PhotonsPositiony + x * PhotonsDirectiony, beta)
    f = eit - (PhotonsPositionz + x * PhotonsDirectionz)
    return f


def Intersectionpoint(PhotonsPosition, PhotonsDirection, M, N, v, t, beta):
    point = scipy.optimize.root(func, 0, args=(PhotonsPosition, PhotonsDirection, M, N, v, t, beta))
    if point.success == False:
        print("Intersectionpoint error")
    return point


def PhotonScatter(PhotonsDirection, gfactor):
    PhotonsDirectionx, PhotonsDirectiony, PhotonsDirectionz = PhotonsDirection
    Carpan = (1 - gfactor * gfactor) / (1 - gfactor + 2 * gfactor * np.random.random())
    thetas = np.arccos((1 + gfactor * gfactor - Carpan * Carpan) / (2 * gfactor))
    phis = 2 * math.pi * np.random.random()
    NextPhotonsDirectionx = np.sin(thetas)*(PhotonsDirectionx*PhotonsDirectionz*np.cos(phis)-PhotonsDirectiony*np.sin(phis))/np.sqrt(1-PhotonsDirectionz**2)+PhotonsDirectionx*np.cos(thetas)
    NextPhotonsDirectiony = np.sin(thetas)*(PhotonsDirectiony*PhotonsDirectionz*np.cos(phis)+PhotonsDirectionx*np.sin(phis))/np.sqrt(1-PhotonsDirectionz**2)+PhotonsDirectiony*np.cos(thetas)
    NextPhotonsDirectionz = -np.sin(thetas)*np.cos(phis)*np.sqrt(1-PhotonsDirectionz**2)+PhotonsDirectionz*np.cos(thetas)
    return np.array([NextPhotonsDirectionx, NextPhotonsDirectiony, NextPhotonsDirectionz])


def MultiProcessPhotons(Indext, Index, PhotonsMatrix, b, c, M, N, v, t, beta, DieEnergy,
                        RefractiveIndexWater, RefractiveIndexAir, gfactor, Radius,
                                            RLocationx, RLocationy, RLocationz, PhiR):
    print(str(Indext) + "  " + str(Index))
    PhotonsPosition = PhotonsMatrix[Index, 0:3]
    PhotonsDirection = PhotonsMatrix[Index, 3:6]
    Wfactor = PhotonsMatrix[Index, 6]
    flag = PhotonsMatrix[Index, 7]
    # underwater
    while flag >= 0:
        if Wfactor < DieEnergy:
            return

        # Zetas = np.random.random()
        Deltas = -1 * np.log(np.random.random()) / c  # a(lambda)
        NextPhotonsPosition = PhotonsPosition + Deltas * PhotonsDirection

        if WaterSurface3D.Watersurface(M, N, v, t, NextPhotonsPosition[0],
                                       NextPhotonsPosition[1], beta) > NextPhotonsPosition[2]:
            PhotonsDirection = PhotonScatter(PhotonsDirection, gfactor)
            PhotonsPosition = NextPhotonsPosition
            Wfactor = Wfactor * b / c
            #flag = flag + 1
        else:
            break

    if flag == -1:
        #PhotonsMatrix[Index, 0:3] = PhotonsPosition
        #PhotonsMatrix[Index, 3:6] = PhotonsDirection
        #PhotonsMatrix[Index, 6] = Wfactor
        #PhotonsMatrix[Index, 7] = flag
        return

    # crosswater
    InitialPointx = Intersectionpoint(PhotonsPosition, PhotonsDirection, M, N, v, t, beta).x
    InitialPoint = PhotonsPosition + InitialPointx * PhotonsDirection
    NormalVector = WaterSurface3D.Watersurfacegradient(M, N, v, t, InitialPoint[0],
                                                       InitialPoint[1], beta)
    Thetaintemp = np.arccos(
        np.dot(NormalVector, PhotonsDirection) / (np.linalg.norm(NormalVector) * np.linalg.norm(PhotonsDirection)))

    if (RefractiveIndexWater / RefractiveIndexAir * np.sin(Thetaintemp))**2 > 1:
        return
    Thetaretemp = np.arcsin(RefractiveIndexWater / RefractiveIndexAir * np.sin(Thetaintemp))
    Angletem = PhotonsDirection - NormalVector * np.dot(NormalVector, PhotonsDirection) / np.linalg.norm(NormalVector)
    NextPhotonsDirection = np.cos(Thetaretemp) * NormalVector + np.sin(Thetaretemp) * Angletem / np.linalg.norm(
        Angletem)
    Ts = np.sin(2 * Thetaintemp) * np.sin(2 * Thetaretemp) / np.sin(Thetaintemp + Thetaretemp) ** 2
    Tre = np.sin(2 * Thetaintemp) * np.sin(2 * Thetaretemp) / (
                np.sin(Thetaintemp + Thetaretemp) ** 2 * np.cos(Thetaintemp - Thetaretemp) ** 2)
    Wfactor = 0.5 * Wfactor * (Ts + Tre)

    if NextPhotonsDirection[0, 2] <= 0:
        return
    else:
        PhotonsDirection = NextPhotonsDirection
        PhotonsPosition = InitialPoint

    #PhotonsMatrix[Index, 0:3] = PhotonsPosition
    #PhotonsMatrix[Index, 3:6] = PhotonsDirection
    #PhotonsMatrix[Index, 6] = Wfactor
    #PhotonsMatrix[Index, 7] = flag

    # receive
    if Wfactor < DieEnergy:
        flag = -1
        return
    else:
        RFlag, ArrivalPosition = ArrivalJudgement(PhotonsPosition, PhotonsDirection.flatten(), RLocationx, RLocationy,
                                                  RLocationz, Radius, PhiR)
        #PhotonsMatrix[Index, 0:3] = ArrivalPosition
        #PhotonsMatrix[Index, 8] = RFlag
        #ArrivalPhotonsPosition = np.concatenate((ArrivalPhotonsPosition, ArrivalPosition[0:2].reshape(1, 2)),
        #                                        axis=0)
        #ArrivalPhotonsPosition = ArrivalPosition[0:2].reshape(1, 2)
        ArrivalPhotonsPositionR = np.empty([0, 2], dtype=np.float64)
        if RFlag != -1:
            #ArrivalPhotonsPositionR = np.concatenate((ArrivalPhotonsPositionR, ArrivalPosition[0:2].reshape(1, 2)),
            #                                         axis=0)
            ArrivalPhotonsPositionR = ArrivalPosition[0:2].reshape(1, 2)
        return ArrivalPhotonsPositionR


if __name__ == '__main__':

    df = pd.read_excel('beta.xlsx', sheet_name=None)
    v = 3
    g = 9.8
    gfactor = 0.924  # clear ocean, coastal, and turbid harbor waters 0.8708, 0.9470,and 0.9199 # 0.924 is a good approximate
    M = 30
    N = 30
    RefractiveIndexWater = 1.3
    RefractiveIndexAir = 1
    Radius = 0.015  # m
    L = 0.03  # m
    x = np.array([-L, 0, L])
    y = np.array([-L, 0, L])
    Locationx, Locationy = np.meshgrid(x, y)
    RLocationx = Locationx.flatten()
    RLocationy = Locationy.flatten()
    ZTx = 1
    ZRx = 1
    RLocationz = ZRx * np.ones(RLocationx.size)
    PhiT = 1  # degree
    PhiR = 90  # degree
    PhotosensitiveRadius = 0.01
    # clear ocean,  a b c
    # coastal,  a, b,  c
    # turbid harbor waters  a, b, c
    ASTable = np.array([[0.114, 0.037, 0.151], [0.179, 0.220, 0.398], [0.366, 1.824, 2.17]])
    # Globalbeta = WaterSurface3D.Randombeta(GlobalM, GlobalN)
    beta = np.array(df.get('Sheet1'))
    DieEnergy = 0.001
    PhotonsNumber = 10**2
    T0 = 0
    Tend = 1
    Tsize = 10
    T = np.linspace(T0, Tend, Tsize)
    # filename = 'Data.xlsx'
    filenameR = 'DataR815_0110102.xlsx'

    # 清空xlsx
    # writer = pd.ExcelWriter(filename)
    writer2 = pd.ExcelWriter(filenameR)
    # DataFRame_data = pd.DataFrame([])
    # DataFRame_data.to_excel(writer, sheet_name=str(0), header=False, index=False)
    DataFRame_data2 = pd.DataFrame([])
    DataFRame_data2.to_excel(writer2, sheet_name=str(0), header=False, index=False)
    # writer.close()  # 关闭，不然打开这个文件只能只读（例如wps打开）
    writer2.close()


    with Pool(24) as pool:
        for Indext in range(Tsize):
            t = T[Indext]
            print(Indext)
            PhotonsMatrix = Photonsgenerater(PhotonsNumber)
            #clear ocean
            b = ASTable[0, 1]
            c = ASTable[0, 2]
            Num = 0
            ##pool = Pool(processes=8)
            ArrivalPhotonsInf = pool.starmap(MultiProcessPhotons, [(Indext, Index,
                                        PhotonsMatrix, b, c, M, N, v, t,
                                        beta, DieEnergy, RefractiveIndexWater, RefractiveIndexAir, gfactor, Radius,
                                                RLocationx, RLocationy, RLocationz, PhiR)
                                                        for Index in range(PhotonsNumber)])
            #pool.close()

            ArrivalPhotonsPositionR = np.empty([0, 2], dtype=np.float64)
            print(ArrivalPhotonsInf)
            for IndexArrivalPhotonsInf in ArrivalPhotonsInf:
                if IndexArrivalPhotonsInf is not None:
                    print(IndexArrivalPhotonsInf)
                    ArrivalPhotonsPositionR = np.concatenate((ArrivalPhotonsPositionR,
                                                              IndexArrivalPhotonsInf), axis=0)



            #print(np.array(ArrivalPhotonsInf).reshape(len(ArrivalPhotonsInf), 2))
        #    for Index in range(PhotonsNumber):
        #       ArrivalPhotonsPosition, ArrivalPhotonsPositionR = MultiProcessPhotons(Indext, Index, PhotonsMatrix, ArrivalPhotonsPosition, ArrivalPhotonsPositionR, b, c, M, N, v, t, beta)

            #writer = pd.ExcelWriter(filename, mode='a', engine='openpyxl',if_sheet_exists='replace')
            writer2 = pd.ExcelWriter(filenameR, mode='a', engine='openpyxl',if_sheet_exists='replace')
            #DataFRame_data = pd.DataFrame(ArrivalPhotonsPosition)
            #DataFRame_data.to_excel(writer, sheet_name=str(Indext), header=False, index=False)
            DataFRame_data2 = pd.DataFrame(ArrivalPhotonsPositionR)
            DataFRame_data2.to_excel(writer2, sheet_name=str(Indext), header=False, index=False)
            #writer.close()  # 关闭，不然打开这个文件只能只读（例如wps打开）
            writer2.close()

       # 关闭进程池，不再接受新的进程
