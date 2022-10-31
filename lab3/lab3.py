# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from skimage.io import show
from scipy.special import erf

var = 1
N = 200
p = 0.3

H = [[0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0]]

T = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0]]


def generateBinVectors(typeVector, N, p):
    arrBinVectors = []
    size = np.shape(typeVector)
    for n in range(0, N):
        binVector = np.zeros(size)
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                u = random.random()
                if u <= p:
                    binVector[i][j] = 1 - typeVector[i][j]
                else:
                    binVector[i][j] = typeVector[i][j]
        arrBinVectors.append(binVector)
    return arrBinVectors


def printVectors(vectors):
    for el in vectors:
        for row in el:
            print(row)
        print("\n")


def getMatPequalOne(vectors):
    size = np.shape(vectors)
    matrixP = np.sum(vectors, axis=0) / size[0]
    return matrixP


def BinaryClassificator(x, class0, class1, P0, P1, names):
    size = np.shape(x)
    w01 = np.zeros(np.shape(x))
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    L = 0
    lymbda = 0
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            w01[i][j] = np.log(p0[i][j] / (1 - p0[i][j]) * (1 - p1[i][j]) / p1[i][j])
            L += x[i][j] * w01[i][j]
            lymbda += np.log((1 - p1[i][j]) / (1 - p0[i][j]))
    lymbda += np.log(P1 / P0)
    if L > lymbda:
        return names[0]
    return names[1]


def calcW01(class0, class1):
    size = np.shape(class1)
    w01 = np.zeros((size[1], size[2]))
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    for i in range(0, size[1]):
        for j in range(0, size[2]):
            w01[i][j] = np.log(p0[i][j] / (1 - p0[i][j]) * (1 - p1[i][j]) / p1[i][j])
    return w01


def createColors(z):
    zmax = np.array(z).max()
    zmin = np.array(z).min()
    colormap = []
    colors = ["b", "m", "r"]
    for Z in z:
        if Z == zmax:
            colormap.append(colors[-1])
        else:
            for n in range(0, len(colors)):
                if (Z >= zmin + n * (zmax - zmin) / len(colors)) & (Z < zmin + (n + 1) * (zmax - zmin) / len(colors)):
                    colormap.append(colors[n])
    return colormap


def binM(class0, class1):
    size = np.shape(class1)
    arrM = [0, 0]
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    for i in range(0, size[1]):
        for j in range(0, size[2]):
            arrM[0] += np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j]) * p0[i][j]
            arrM[1] += np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j]) * p1[i][j]
    return arrM


def binD(class0, class1):
    size = np.shape(class1)
    arrD = [0, 0]
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    for i in range(0, size[1]):
        for j in range(0, size[2]):
            arrD[0] += np.square(np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j])) * p0[i][j] * (
                        1 - p0[i][j])
            arrD[1] += np.square(np.log(p1[i][j] / (1 - p1[i][j]) * (1 - p0[i][j]) / p0[i][j])) * p1[i][j] * (
                        1 - p1[i][j])
    return arrD


def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def calcErrors(class0, class1, P0, P1):
    size = np.shape(class0[0])
    p = [0, 0]
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    lymbda = 0
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            lymbda += np.log((1 - p1[i][j]) / (1 - p0[i][j]))
    lymbda += np.log(P1 / P0)
    M = binM(class0, class1)
    D = binD(class0, class1)
    p[0] = 1 - Phi((lymbda - M[0]) / np.sqrt(D[0]))
    p[1] = Phi((lymbda - M[1]) / np.sqrt(D[1]))
    return p


def calcLyambda(class0, class1, P0, P1):
    size = np.shape(class0[0])
    p0 = getMatPequalOne(class0)
    p1 = getMatPequalOne(class1)
    lymbda = 0
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            lymbda += np.log((1 - p1[i][j]) / (1 - p0[i][j]))
    lymbda += np.log(P1 / P0)
    return lymbda


def experimentErrors(expClass, classes, arrP, className, names, pTheoretic):
    expP = 0
    for vector in expClass:
        if BinaryClassificator(vector, classes[0], classes[1], arrP[0], arrP[1], names) != className:
            expP += 1
    print(f"amount of invalid vectors {className}: {expP}")
    expP /= len(expClass)
    if expP != 0:
        e = np.sqrt((1 - pTheoretic) / (expP * len(expClass)))
    else:
        e = 0
    return expP, e


def findInvalidVector(expClass, classes, arrP, className, names):
    invalidClass = []
    for vector in expClass:
        if BinaryClassificator(vector, classes[0], classes[1], arrP[0], arrP[1], names) != className:
            invalidClass.append(vector)
    if len(invalidClass) < 1:
        invalidClass.append(np.zeros_like(expClass[0]))
    return invalidClass


def printVectors(vector, fig, pos):
    ax1 = fig.add_subplot(pos, projection='3d')

    size = np.shape(vector)
    _x = np.arange(size[0])
    _y = np.arange(size[1])
    _xx, _yy = np.meshgrid(_x, _y)  # создание матрицы индексов х и у
    x, y = _xx.ravel(), _yy.ravel()
    z = []
    for k in range(0, len(x)):
        z.append(vector[x[k]][y[k]])
    bottom = np.zeros_like(z)
    width = depth = 1

    zmin = np.array(z).min() - 1
    zmax = np.array(z).max() + 1
    ax1.set_xlim3d(0, size[0])
    ax1.set_ylim3d(0, size[1])
    ax1.set_zlim3d(zmin, zmax)

    color = createColors(z)
    a = ax1.bar3d(x, y, bottom, width, depth, z, shade=True, color=color, edgecolor='black')  # , cmap='inferno')
    # fig.colorbar(a, ticks=np.arange(0.0, 1.0, (zmax-1)/3))
    return fig


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # task 1.5
    vectorsH = generateBinVectors(H, N, p)
    vectorsT = generateBinVectors(T, N, p)

    # task 2.4
    Pij_H = getMatPequalOne(vectorsH)
    Pij_T = getMatPequalOne(vectorsT)

    classOfVector = BinaryClassificator(vectorsH[0], vectorsH, vectorsT, 0.5, 0.5, ["class H", "class T"])
    print(classOfVector)
    classOfVector = BinaryClassificator(vectorsT[0], vectorsH, vectorsT, 0.5, 0.5, ["class H", "class T"])
    print(classOfVector)

    arrM = binM(vectorsH, vectorsT)
    arrD = binD(vectorsH, vectorsT)
    limbda = calcLyambda(vectorsH, vectorsT, 0.5, 0.5)

    w = calcW01(vectorsH, vectorsT)

    errs = calcErrors(vectorsH, vectorsT, 0.5, 0.5)
    print(f"p for class H: {errs[0]:.8f}\np for class T: {errs[1]:.8f}")
    print("\n")
    errsExperiment = experimentErrors(vectorsH, [vectorsH, vectorsT], [0.5, 0.5], "H", ["H", "T"], errs[0])
    print(f"experiment p for class H: {errsExperiment[0]:.6f}\ne: {errsExperiment[1] * 100:.6f}%")
    errsExperimentT = experimentErrors(vectorsT, [vectorsH, vectorsT], [0.5, 0.5], "T", ["H", "T"], errs[1])
    print(f"experiment p for class T: {errsExperimentT[0]:.6f}\ne: {errsExperimentT[1] * 100:.6f}%")

    invalidH = findInvalidVector(vectorsH, [vectorsH, vectorsT], [0.5, 0.5], "H", ["H", "T"])
    invalidT = findInvalidVector(vectorsT, [vectorsH, vectorsT], [0.5, 0.5], "T", ["H", "T"])

    x1 = np.linspace(arrM[0] - 3 * np.sqrt(arrD[0]), arrM[0] + 3 * np.sqrt(arrD[0]), 100)
    x2 = np.linspace(arrM[1] - 3 * np.sqrt(arrD[1]), arrM[1] + 3 * np.sqrt(arrD[1]), 100)
    min1 = np.min(x1)
    min2 = np.min(x2)
    max1 = np.max(x1)
    max2 = np.max(x2)
    x = np.linspace(np.min([min1, min2]), np.max([max1, max2]), 200)
    fig0 = plt.figure(figsize=(10, 10))
    plt.plot(x, scipy.stats.norm.pdf(x, arrM[0], np.sqrt(arrD[0])))
    plt.plot(x, scipy.stats.norm.pdf(x, arrM[1], np.sqrt(arrD[1])))
    # plt.axvline(x=limbda, c='y')

    fig = plt.figure(figsize=(10, 10))
    fig = printVectors(H, fig, 221)
    fig = printVectors(T, fig, 222)
    fig = printVectors(Pij_H, fig, 223)
    fig = printVectors(Pij_T, fig, 224)

    fig1 = plt.figure(figsize=(10, 10))
    fig1 = printVectors(w, fig1, 121)
    fig1 = printVectors(-w, fig1, 122)

    fig2 = plt.figure(figsize=(10, 10))
    fig2 = printVectors(H, fig2, 321)
    fig2 = printVectors(T, fig2, 322)
    fig2 = printVectors(vectorsH[0], fig2, 323)
    fig2 = printVectors(vectorsT[0], fig2, 324)
    fig2 = printVectors(invalidH[0], fig2, 325)
    fig2 = printVectors(invalidT[0], fig2, 326)
    show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
