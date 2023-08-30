import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Radius = 0.015 #m
L = 0.03 #m
T0 = 0
Tend = 10
Tsize = 100
scale = 1
x = np.array([-L, 0, L])
y = np.array([-L, 0, L])
Locationx, Locationy = np.meshgrid(x, y)
RLocationx = Locationx.flatten()
RLocationy = Locationy.flatten()
filenamelist = ["DataRB813_010100105",
                "DataFA816_1020100105",
                "DataFB816_2030100105",
                "DataFC816_3040100105",
                "DataRB817_4050100105",
                "DataFA817_5060100105",
                "DataFB817_6070100105",
                "DataFC817_7080100105",
                "DataFA818_80100200105",
                "DataRB823_80100100105B",
                "DataRB825_100110100105",
                "DataRB827_1075110100105"
                ]


FileSize = []
FileSizeBase = []
for filenameIndex in range(len(filenamelist)):
    df = pd.ExcelFile(filenamelist[filenameIndex] + '.xlsx')
    sheet_names = df.sheet_names
    FileSizeBase.append(int(np.sum(FileSize[0:filenameIndex])))
    FileSize.append(len(sheet_names))

print(FileSize)

Num = np.zeros((np.sum(FileSize), 10))
T = np.linspace(0, 0.1*np.sum(FileSize), np.sum(FileSize))
for fileIndex in range(len(filenamelist)):
    df = pd.read_excel(filenamelist[fileIndex]+'.xlsx', sheet_name=None, header=None)
    for IndexSheetName in range(FileSize[fileIndex]):
        PhotonsLocationx = np.array(df.get(str(IndexSheetName)))[:, 0]
        PhotonsLocationy = np.array(df.get(str(IndexSheetName)))[:, 1]
        for Index in range(PhotonsLocationx.size):
            xp = PhotonsLocationx[Index]
            yp = PhotonsLocationy[Index]
            for IndexR in range(RLocationx.size):
                xRx = RLocationx[IndexR]
                yRx = RLocationy[IndexR]
                flag = (xp - xRx) ** 2 + (yp - yRx) ** 2 < Radius ** 2
                if flag == 1:
                    Num[IndexSheetName+FileSizeBase[fileIndex], IndexR] = (
                            Num[IndexSheetName+FileSizeBase[fileIndex], IndexR] + 1)
        Num[IndexSheetName+FileSizeBase[fileIndex], 9] = np.sum(Num[IndexSheetName+FileSizeBase[fileIndex], 0:9])

DataFRame_data = pd.DataFrame(Num)
DataFRame_data.to_excel('AllDataProcess.xlsx', sheet_name=str(0), header=False, index=False)



plt.subplot(911)
plt.plot(T, Num[:, 0])
plt.subplot(912)
plt.plot(T, Num[:, 1])
plt.subplot(913)
plt.plot(T, Num[:, 2])
plt.subplot(914)
plt.plot(T, Num[:, 3])
plt.subplot(915)
plt.plot(T, Num[:, 4])
plt.subplot(916)
plt.plot(T, Num[:, 5])
plt.subplot(917)
plt.plot(T, Num[:, 6])
plt.subplot(918)
plt.plot(T, Num[:, 7])
plt.subplot(919)
plt.plot(T, Num[:, 8])
plt.savefig('MultiData.png', dpi=600)
plt.show(block=True)