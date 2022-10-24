# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy import double
from skimage.io import imsave, imshow, show, imread
from scipy.special import erf, erfinv
import math

var = 1
M1 = [0, 0]
M2 = [1, 1]
M3 = [-1, 1]
B1 = [[0.11, 0.0],
      [0.0, 0.19]]
# B1 = [[0.02, 0.0],
#       [0.0, 0.02]]
B2 = [[0.23, 0.01],
      [0.02, 0.17]]
B3 = [[0.2, 0.1],
      [0.1, 0.3]]

n = 2
N = 200

# lab 1
def generate_vectors(A, M, n, N):
    left_border = 0
    right_border = 1
    m = (right_border + left_border) / 2
    k = 20
    Sn = np.zeros((n, N))
    for i in range(0, k, 1):
        Sn += np.random.uniform(left_border, right_border, (n, N)) - m
    # Sn = s1 + s2 + ... + sn - m-m-m-m-m -> (s1-m) + (s2-m) + ... + (sn-m)
    CKO = (right_border - left_border) / np.sqrt(12)
    # CKO = sqrt(D)  D = (b-a)^2/12
    E = Sn / (CKO * np.sqrt(k))
    x = np.matmul(A, E) + np.reshape(M, (2, 1)) * np.ones((1, N))
    return x


def calcMatrixA(B):
    A = np.zeros((2, 2))
    A[0][0] = np.sqrt(B[0][0])
    A[0][1] = 0
    A[1][0] = B[0][1] / np.sqrt(B[0][0])
    A[1][1] = np.sqrt(B[1][1] - (B[0][1] ** 2) / B[0][0])
    return A


def M(x):
    M = np.sum(x, axis=1) / N
    return M


def rateB(x, M):
    B = np.zeros((2, 2))
    for i in range(0, N, 1):
        tmp = np.reshape(x[:, i], (2, 1))
        B += (np.matmul(tmp, np.transpose(tmp)))
    B /= N
    B -= np.matmul(np.reshape(M, (2, 1)), np.reshape(M, (1, 2)))
    return B

#lab 2
# фугкция определения вектора к классу среди нескольких классов (Байес)
def BayeslassificatorB(x, arrM, arrB, L):
    # создать матрицу истиности для dij и записывать true, если d больше либо 0, иначе false
    # строка со всеми true и будет определять класс своим индексом
    X = np.reshape(x, (2, 1))
    table = np.eye(len(arrM)).astype(bool)

    flag = True
    for el in arrB:
        if el != arrB[0]:
            flag = False

    if flag==True:
        # одинаковые В
        for i in range(0, len(arrM)-1):
            for j in range(i+1, len(arrM)):
                difM = np.reshape(arrM[i], (1, 2)) - np.reshape(arrM[j], (1, 2))
                sumM = np.reshape(arrM[i], (1, 2)) + np.reshape(arrM[j], (1, 2))
                tmp1 = np.matmul(np.matmul(difM, np.linalg.inv(arrB[0])), X)
                tmp2 = 0.5 * np.matmul(np.matmul(sumM, np.linalg.inv(arrB[0])), np.transpose(difM))
                dij = tmp1 - tmp2 + np.log(L)
                if dij >= 0:
                    table[i][j] = True
                    table[j][i] = False
                else:
                    table[i][j] = False
                    table[j][i] = True
        # print(table)
        for i in range(0, len(table)):
            if (table[i] == np.ones(len(table)).astype(bool)).all():
                return i
    else:
        # разные В
        for i in range(0, len(arrM)-1):
            for j in range(i+1, len(arrM)):
                divB = math.log(np.linalg.det(arrB[i]) / np.linalg.det(arrB[j]))
                dist1 = np.matmul(np.matmul(np.reshape(arrM[i], (1, 2)), np.linalg.inv(arrB[i])), np.reshape(arrM[i], (2, 1)))
                dist2 = np.matmul(np.matmul(np.reshape(arrM[j], (1, 2)), np.linalg.inv(arrB[j])), np.reshape(arrM[j], (2, 1)))
                tmp1 = np.matmul(np.matmul(np.transpose(X), (np.linalg.inv(arrB[j]) - np.linalg.inv(arrB[i]))), X)
                tmp2 = 2*(np.matmul(np.reshape(arrM[i], (1, 2)), np.linalg.inv(arrB[i])) -
                          np.matmul(np.reshape(arrM[j], (1, 2)), np.linalg.inv(arrB[j])))
                tmp2 = np.matmul(tmp2, X)
                tmp3 = divB + 2*np.log(L) - dist1 + dist2
                dij = tmp1 + tmp2 + tmp3
                if dij >= 0:
                    table[i][j] = True
                    table[j][i] = False
                else:
                    table[i][j] = False
                    table[j][i] = True
        # print(table)
        for i in range(0, len(table)):
            if (table[i] == np.ones(len(table)).astype(bool)).all():
                return i

# функция возвращает график(маасив х, массив у) Байесовской границы для одинаковых В
def BayesBorderSampleB(m1, m2, b, L):
    # исходя из уравнения dij = (Mi-Mj) * inv(B) * (x, y) - 0.5(Mi+Mj)*inv(B)(Mi-Mj) + Ln( P(i)/P(j) )
    # получаем 0 = (a, b)*(x, y) + (d) + ln => [(d) + ln = c] => ax + by + c = 0
    # вырахаем у через х --> получили график границы
    # в функции просто зашита формула dij и по ней расчитан график
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    sumM = np.reshape(m1, (1, 2)) + np.reshape(m2, (1, 2))
    tmp1 = np.matmul(sumM, np.linalg.inv(b))
    tmp2 = 0.5 * np.reshape(np.matmul(tmp1, np.reshape(difM, (2, 1))), (1, )) - np.log(L)
    tmp3 = np.reshape(np.matmul(difM, np.linalg.inv(b)), (2, ))
    k = -tmp3[0]/tmp3[1]
    c = tmp2[0]/tmp3[1]
    x = np.arange(-3, 3, 0.1)
    y = k*x + c
    return x, y

# интегральная функция нормального распределения через функцию ошибки
def Phi(x):
    return 0.5*(1 + erf(x/np.sqrt(2)))

# обратная интегральная функция нормального распределения через обратную функцию ошибки
def invPhi(x):
    return np.sqrt(2)*erfinv(2*x-1)

# подаются 2 класса с одинаковыми В
# считаются теоретические вероятности ошибочной классификации
def p_error(m1, m2, b):
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    MahalnobisDistant = np.reshape(np.matmul(np.matmul(difM, np.linalg.inv(b)), np.transpose(difM)), (1,))
    p = [0, 0]
    p[0] = 1 - Phi(0.5*MahalnobisDistant[0]/np.sqrt(MahalnobisDistant[0]))
    p[1] = Phi(-0.5*MahalnobisDistant[0]/np.sqrt(MahalnobisDistant[0]))
    return p

# функция возвращает график(маасив х, массив у) Байесовской границы для разных В
def BayesBorderDifferenceB(m1, m2, b1, b2, x):
    # исходя из уравнения dij = (x,y) * inv(Bj-Bi) * (x,y) + 2( Mi*inv(Bi)-Mj*inv(Bj) )*(x,y) + Ln(|inv(Bi)|/|inv(Bj)|)+
    # + 2Ln( P(i)/P(j) ) - Mi*inv(Bi)*Mi + Mj*inv(Bj)*Mj
    # после раскрытия всего получим
    # получаем 0 = (x, y)*({a, b}, {c, d})*(x, y) + (e, f)*x + g =>
    # ax^2 + (b+c)xy + dy^2 + ex + fy + g = 0 - это кривая второго порядка(параболла/эллипс/гипербола)
    # решаем квадратное уравнение относительно у
    # ищем дискриминант D
    # получим y1 = (-(c+b)x + f + sqrt(D))/2d
    # получим y2 = (-(c+b)x + f - sqrt(D))/2d
    # получили уравнения двух веток для графика. Готово.
    divB = math.log(np.linalg.det(b2)/np.linalg.det(b1))
    dist1 = np.matmul(np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b1)), np.reshape(m1, (2, 1)))
    dist2 = np.matmul(np.matmul(np.reshape(m2, (1, 2)), np.linalg.inv(b2)), np.reshape(m2, (2, 1)))
    a = np.linalg.inv(b2) - np.linalg.inv(b1)
    b = np.reshape(2*(np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b1))-np.matmul(np.reshape(m2, (1, 2)),
                                                                                     np.linalg.inv(b2))), (2,))
    c = np.reshape((divB - dist1 + dist2), (1,))
    D = (np.square((a[1][0] + a[0][1]))-4*a[0][0]*a[1][1])*np.square(x) + \
        (2*(a[1][0] + a[0][1])*b[1] - 4*b[0]*a[1][1])*x + b[1]*b[1] - 4*a[1][1]*c[0]
    # print(np.sqrt(D))
    y1 = (-(a[1][0] + a[0][1]) * x - b[1] + np.sqrt(D)) / (2*a[1][1])
    y2 = (-(a[1][0] + a[0][1]) * x - b[1] - np.sqrt(D)) / (2*a[1][1])
    return x, y1, y2

# функция находит порог лямбда для минимаксного классификатора с точностью не больше Е
# точно вычислить невозможно, так как основная функция не является стандартной, поэтому ищется приблизительное значение
def findL(m1, m2, b, c0, c1, E):
    L = c1 / c0
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    p = np.reshape(np.matmul(np.matmul(difM, b), np.transpose(difM)), (1,))
    F = c0 / c1 * Phi((-(np.log(L * c0 / c1)) - 0.5 * p[0]) / np.sqrt(p[0])) - Phi(
        ((np.log(L * c0 / c1)) - 0.5 * p[0]) / np.sqrt(p[0]))

    while np.abs(F) > E:
        if F < 0:
            L /= 2
        else:
            L *= 3
        F = c0 / c1 * Phi((-(np.log(L * c0 / c1)) - 0.5 * p[0]) / np.sqrt(p[0])) - \
            Phi(((np.log(L * c0 / c1)) - 0.5 * p[0]) / np.sqrt(p[0]))
    return L

# Минимаксный классификатор
# пользуемся тем, что он является Байессовским при определенных соотношениях P(0) и P(1)
def classificatorMinMax(x, m1, m2, b, c0, c1):
    L = findL(m1, m2, b, c0, c1, 0.000001)
    arrM = [m1, m2]
    arrB = [b, b]
    return BayeslassificatorB(x, arrM, arrB, L)

# граница минимаксного классификатора
# пользуемся тем, что он является Байессовским, поэтому строим границу как для Байесовского
def borderMinMax(m1, m2, b, c0, c1):
    L = findL(m1, m2, b, c0, c1, 0.000001)
    return BayesBorderSampleB(m1, m2, b, L)

# Нейман-Пирс классификатор
# просто происходит сравнение f(x|1)/f(x|0) и лямбды, из чего вектор относится к тому или иному классу
def classificatorNP(x, p0, m1, m2, b):
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    tmp0 = np.matmul(difM, np.linalg.inv(b))
    distMahal = np.matmul(tmp0, np.transpose(difM))
    ro = distMahal[0][0]
    lymbda = -0.5 * ro + np.sqrt(ro) * invPhi(1 - p0)
    l = np.exp(lymbda)
    tmp1 = np.matmul(np.reshape(x, (2, 1)) - np.reshape(m1, (1, 2)), np.linalg.inv(b))
    r1 = np.matmul(tmp1, np.reshape(x, (2, 1)) - np.reshape(m1, (2, 1)))
    tmp2 = np.matmul(np.reshape(x, (2, 1)) - np.reshape(m2, (1, 2)), np.linalg.inv(b))
    r2 = np.matmul(tmp2, np.reshape(x, (2, 1)) - np.reshape(m2, (2, 1)))
    y = np.exp(-0.5*(r2[0][0]-r1[0][0]))
    if (y > l): return 1
    return 0

# граница Неймана-Пирса
# распишем f(x|1)/f(x|0) и сравним с лямбдой
# лямбда рассчитывается из методички как e^( -0.5*ro(M0, M1) + sqrt( ro(M0, M1) )*invPhi(1-p0) ), где r(M0, M1) - расстояние Махаланобиса
# f(x|1)/f(x|0) = sqrt( |B0|/|B1|) * e^( -0.5*[(x-M1)*inv(B1)(x-M1) - (x-M0)*inv(B0)(x-M0)] )
# решая f(x|1)/f(x|0) = лямбда получим уравнение вида: (x,y)*(a,b) + (c,d)*(x,y) + t = 0
# выразим у через х и получим: y = (-(a+c) * x - t)/(b+d)
def borderNPclass(p0, m1, m2, b):
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    tmp1 = np.matmul(difM, np.linalg.inv(b))
    distMahal = np.matmul(tmp1, np.transpose(difM))
    ro = distMahal[0][0]
    lymbda = -0.5*ro + np.sqrt(ro)*invPhi(1-p0)
    difM = np.reshape(m1, (2, 1)) - np.reshape(m2, (2, 1))
    sumM = np.reshape(m1, (2, 1)) + np.reshape(m2, (2, 1))
    a = np.matmul(np.linalg.inv(b), difM)
    d = np.matmul(np.transpose(difM), np.linalg.inv(b))
    tmp2 = np.matmul(np.transpose(difM), np.linalg.inv(b))
    c = -np.matmul(tmp2, sumM)
    k = -(a[0][0]+d[0][0])/(a[1][0]+d[0][1])
    t = -(c[0][0]+2*lymbda)/(a[1][0]+d[0][1])
    x = np.arange(-3, 3, 0.1)
    y = k * x + t
    return x, y


def num2Classname(n, names):
    return names[n]


def classificationError(x, arrM, arrB, p, className, names):
    absError = 0.0
    sizeX = np.shape(x)
    for i in range(0, sizeX[1]):
        n = BayeslassificatorB(x[:, i], arrM, arrB, 1)
        if className != num2Classname(n, names):
            absError += 1
    print(f"amount of {className} in other class is {int(absError)}")
    absError = absError/sizeX[1]
    e = np.sqrt((1-p)/(sizeX[1]*absError))
    return absError, e


def amountVectorsWithError(e, p, expErr):
    N = np.ceil((1-p)/(expErr*e*e))
    return N


# lab4
def calcFishersParametrs(m0, m1, b0, b1):
    difM = np.reshape(m1, (2, 1)) - np.reshape(m0, (2, 1))
    sumB = 0.5*(np.array(b0) + np.array(b1))
    W = np.matmul(np.linalg.inv(sumB), difM)  # size(2, 1)
    D0 = np.matmul(np.matmul(np.transpose(W), b0), W)
    D1 = np.matmul(np.matmul(np.transpose(W), b1), W)
    D0 = D0[0, 0]
    D1 = D1[0, 0]
    tmp = np.matmul(np.transpose(difM), np.linalg.inv(sumB))
    tmp = np.matmul(tmp, (D1*np.reshape(m0, (2, 1)) + D0*np.reshape(m1, (2, 1))))
    wn = -tmp[0, 0]/(D0 + D1)
    return np.reshape(W, (2, )), wn


def calcMSEParameters(class0, class1):
    W = 0
    size1 = np.shape(class1)
    size0 = np.shape(class0)
    z1Size = ((size1[0] + 1), size1[1])
    z0Size = ((size0[0] + 1), size0[1])
    z1 = np.ones(z1Size)
    z0 = np.ones(z0Size)
    z1[0:size1[0], 0:size1[1]] = class1
    z0[0:size0[0], 0:size0[1]] = class0
    z0 = -1*z0

    resSize = (3, (size1[1] + size0[1]))
    z = np.ones(resSize)
    z[0:3, 0:z1Size[1]] = z1
    z[0:3, z1Size[1]:resSize[1]] = z0  # size(3, 400)

    tmp = np.linalg.inv(np.matmul(z, np.transpose(z)))
    R = np.ones((resSize[1], 1))
    W = np.matmul(np.matmul(tmp, z), R)
    return np.reshape(W, (3, ))


def calcACRParameters(initVectors):
    arrW = []
    W = 0
    size = np.shape(initVectors)
    W = np.ones((size[0]-1, ))
    arrW.append(W)
    cnt = 0
    flagSNG = 0
    for k in range(0, size[1]):
        x = np.reshape(initVectors[0:-1, k], (1, size[0]-1))
        r = initVectors[-1, k]
        d = np.matmul(arrW[-1], np.transpose(x))
        if ((d[0] < 0) & (r > 0)) | ((d[0] > 0) & (r < 0)):
            sgn = np.sign(r - d)
            if sgn[0] != flagSNG:
                cnt += 1
                flagSNG = sgn[0]
            W = W + pow(cnt, -0.6)*x*sgn
            arrW.append(np.reshape(W, (size[0]-1,)))
    return arrW


def borderLinClassificator(W, wn, x, nameClassificator):
    # 0 = W0*x + W1*y + wn -> y = -(W0*x + wn)/W1

    # print(f"{nameClassificator}: W: {W}, wn: {wn}")
    if W[1] != 0:
        y = -(W[0]*x + wn)/W[1]
    else:
        x = -(wn/W[0])*np.ones(len(x))
        y = np.linspace(-100, 100, len(x))
    return x, y


def printClassificator(fig, pos, class0, class1, dBayess, dAnother, nameAnotherBorder, lineFormat):
    fig.add_subplot(pos)
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1.5, 2.5)
    # plt.xlim(-30, 30)
    # plt.ylim(-30, 30)
    plt.plot(class0[0], class0[1], 'r+')
    plt.plot(class1[0], class1[1], 'bx')
    plt.plot(dAnother[0], dAnother[1], 'm-')
    c = ['r', 'y', 'g', 'c', 'b', 'm']
    for i in range(1, len(dBayess)):
        plt.plot(dBayess[0], dBayess[i], f'{c[i%6]}{lineFormat}')
    plt.legend(["class Red", "class Blue", nameAnotherBorder+" border", "Bayess border"])
    return fig


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # task 1.2
    # A = calcMatrixA(B1)
    #
    # x1 = generate_vectors(A, M1, n, N)
    # y1 = generate_vectors(A, M2, n, N)

    # fig = plt.figure(figsize=(10, 10))
    # plt.xlim(-2, 3)
    # plt.ylim(-2, 3)
    # plt.plot(x1[0], x1[1], 'r.')
    # plt.plot(y1[0], y1[1], 'b*')

    # task 1.3
    # A1 = calcMatrixA(B1)
    # A2 = calcMatrixA(B2)
    # A3 = calcMatrixA(B3)
    #
    # x2 = generate_vectors(A1, M1, n, N)
    # y2 = generate_vectors(A2, M2, n, N)
    # z2 = generate_vectors(A3, M3, n, N)

    # # fig1 = plt.figure(figsize=(10, 10))
    # # plt.xlim(-3, 3)
    # # plt.ylim(-2, 4)
    # # plt.plot(x2[0], x2[1], 'r.')
    # # plt.plot(y2[0], y2[1], 'gx')
    # # plt.plot(z2[0], z2[1], 'b+')
    # # show()
    #
    # # task 1.4
    # Mx1 = M(x1)
    # My1 = M(y1)
    # Mx2 = M(x2)
    # My2 = M(y2)
    # Mz2 = M(z2)
    #
    # Bx1 = rateB(x1, Mx1)
    # By1 = rateB(y1, My1)
    # Bx2 = rateB(x2, Mx2)
    # By2 = rateB(y2, My2)
    # Bz2 = rateB(z2, Mz2)
    #
    # print(f'M1: {M1} <-- [{Mx1[0]:.3f}, {Mx1[1]:.3f}]')
    # print(f"M1: {M2} <-- [{My1[0]:.3f}, {My1[1]:.3f}]")
    # print(f"M1: {M1} <-- [{Mx2[0]:.3f}, {Mx2[1]:.3f}]")
    # print(f"M2: {M2} <-- [{My2[0]:.3f}, {My2[1]:.3f}]")
    # print(f"M3: {M3} <-- [{Mz2[0]:.3f}, {Mz2[1]:.3f}]\n")
    # print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{Bx1[0][0]:.3f}, {Bx1[0][1]:.3f}],")
    # print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{Bx1[1][0]:.3f}, {Bx1[1][1]:.3f}]\n")
    # print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{By1[0][0]:.3f}, {By1[0][1]:.3f}],")
    # print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{By1[1][0]:.3f}, {By1[1][1]:.3f}]\n")
    # print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{Bx2[0][0]:.3f}, {Bx2[0][1]:.3f}],")
    # print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{Bx2[1][0]:.3f}, {Bx2[1][1]:.3f}]\n")
    # print(f"B2: [{B2[0][0]:.2f}, {B2[0][1]:.2f}],   <--  [{By2[0][0]:.3f}, {By2[0][1]:.3f}],")
    # print(f"    [{B2[1][0]:.2f}, {B2[1][1]:.2f}]         [{By2[1][0]:.3f}, {By2[1][1]:.3f}]\n")
    # print(f"B3: [{B3[0][0]:.2f}, {B3[0][1]:.2f}],   <--  [{Bz2[0][0]:.3f}, {Bz2[0][1]:.3f}],")
    # print(f"    [{B3[1][0]:.2f}, {B3[1][1]:.2f}]         [{Bz2[1][0]:.3f}, {Bz2[1][1]:.3f}]")



    # print(f"\n\n\n\n\nLab2:\n")
    # # task 2.1
    # # x1 и y1 - байессовская граница
    # x = [-1, 0]
    # arrM = [M1, M2]
    # arrB = [B1, B1]
    # classN = BayeslassificatorB(x, arrM, arrB, 1)
    # className = num2Classname(classN, ["red", "blue"])
    # print(f"Bayess class: {className}")
    #
    # d1 = BayesBorderSampleB(M1, M2, B1, 1)
    # p = p_error(M1, M2, B1)
    # R = p[0]
    # print(f"Ошибка классификации в первый класс: {p[0]:.3f}\n"
    #       f"Ошибка классификации во второй класс: {p[1]:.3f}\n"
    #       f"Общий риск классификации: {p[0]:.3f}\n")
    #
    #
    # # task 2.2
    # # x1 и y1 - классификаторы минимаксный и Неймана-Пирсона
    # # минимаксный
    # d2 = borderMinMax(M1, M2, B1, 2, 2)
    # classN = classificatorMinMax(x, M1, M2, B1, 2, 2)
    # className = num2Classname(classN, ["red", "blue"])
    # print(f"Minmax class: {className}")
    #
    # # Неймана-Пирсона
    # p0 = 0.05
    # d3 = borderNPclass(p0, M1, M2, B1)
    # classN = classificatorNP(x, p0, M1, M2, B1)
    # className = num2Classname(classN, ["red", "blue"])
    # print(f"NP class: {className}")
    #
    # fig2 = plt.figure(figsize=(10, 10))
    # plt.xlim(-2, 3)
    # plt.ylim(-2, 3)
    # plt.plot(x1[0], x1[1], 'r.')
    # plt.plot(y1[0], y1[1], 'b*')
    # plt.plot(d1[0], d1[1], 'm-')
    # plt.plot(d2[0], d2[1], 'y.')
    # plt.plot(d3[0], d3[1], 'c--')
    # plt.plot(x[0], x[1], 'k*')
    # plt.legend(["class Red", "class Blue", "Bayes border", "Minmax border", "NP border", "point"])
    #
    #
    # # task 2.3
    # # x2, y2 и z2 - байессовская граница
    # # обрезание картинки производилось исходя из полных графиков
    # # посмотрел примерные координаты пересечения и подставил их в ограничения t1 и t2
    # t1 = np.arange(-0.0996792, 3, 0.000001)
    # t2 = np.arange(-3, -0.0996139, 0.000001)
    # dxy = BayesBorderDifferenceB(M1, M2, B1, B2, t1)
    # dxz = BayesBorderDifferenceB(M1, M3, B1, B3, t2)
    # dyz = BayesBorderDifferenceB(M2, M3, B2, B3, t1)
    #
    # arrM = [M1, M2, M3]
    # arrB = [B1, B2, B3]
    # classN = BayeslassificatorB(x, arrM, arrB, 1)
    # className = num2Classname(classN, ["red", "green", "blue"])
    # print(f"Bayess class number: {className}")
    # # найти пересечение и обрезать границы
    #
    # errsX = classificationError(x1, [M1, M2], [B1, B1], p[0], "red", ["red", "blue"])
    # errsY = classificationError(y1, [M1, M2], [B1, B1], p[1], "blue", ["red", "blue"])
    # print(f"p experiment for class x1: {errsX[0]:.4f}")
    # print(f"e for class x1: {errsX[1]:.4f}\n")
    # print(f"p experiment for class y1: {errsY[0]:.4f}")
    # print(f"e for class y1: {errsY[1]:.4f}\n")
    #
    # e = 0.05
    # Nx = amountVectorsWithError(e, p[0], errsX[0])
    # Ny = amountVectorsWithError(e, p[1], errsY[0])
    # print(f"size class x1 for e<{e} : {Nx}")
    # print(f"size class y1 for e<{e} : {Ny}")
    #
    #
    # fig4 = plt.figure(figsize=(10, 10))
    # plt.xlim(-3, 3)
    # plt.ylim(-2, 4)
    # plt.plot(x[0], x[1], 'k*')
    # plt.plot(x2[0], x2[1], 'r.')
    # plt.plot(y2[0], y2[1], 'gx')
    # plt.plot(z2[0], z2[1], 'b+')
    # # plt.plot(dxy[0], dxy[1], 'y-') # в экран не влазит, можно не показывать
    # plt.plot(dxy[0], dxy[2], 'y-')
    # # plt.plot(dxz[0], dxz[1], 'c-')
    # plt.plot(dxz[0], dxz[2], 'c-')
    # # plt.plot(dyz[0], dyz[1], 'm-')
    # plt.plot(dyz[0], dyz[2], 'm-')
    # plt.legend(["point", "class Red", "class Green", "class Blue", "Border RedvsGreen", "Border RedvsBlue", "Border GreenvsBlue"])
    # show()

    print(f"\n\n\n\n\nLab4:\n")
    # lab 4
    A1 = calcMatrixA(B1)
    A2 = calcMatrixA(B2)

    x3 = generate_vectors(A1, M1, n, N)
    y3 = generate_vectors(A2, M2, n, N)
    z3 = generate_vectors(A1, M2, n, N)

    sizeX = np.shape(x3)
    sizeY = np.shape(y3)
    sizeZ = np.shape(z3)
    # task 4.1
    # Классификатор, максимизирующий критерий Фишера
    W1, wn1 = calcFishersParametrs(M1, M2, B1, B1)
    W2, wn2 = calcFishersParametrs(M1, M2, B1, B2)

    t3 = np.linspace(-2, 3, 100)
    dBayess = BayesBorderSampleB(M1, M2, B1, 1)
    dFisher = borderLinClassificator(W1, wn1, t3, "Fisher with sample B")

    dBayess2 = BayesBorderDifferenceB(M1, M2, B1, B2, t3)
    dFisher2 = borderLinClassificator(W2, wn2, t3, "Fisher with different B")

    fig5 = plt.figure(figsize=(16, 7))
    fig5 = printClassificator(fig5, 121, x3, z3, dBayess, dFisher, "Fisher", ".")
    fig5 = printClassificator(fig5, 122, x3, y3, dBayess2, dFisher2, "Fisher", "-")


    # task 4.2
    # Классификатор, минимизирующий СКО
    Wmse1 = calcMSEParameters(x3, z3)
    dMSE1 = borderLinClassificator(Wmse1[0:2], Wmse1[-1], t3, "MSE with sample B")

    Wmse2 = calcMSEParameters(x3, y3)
    dMSE2 = borderLinClassificator(Wmse2[0:2], Wmse1[-1], t3, "MSE with different B")

    fig6 = plt.figure(figsize=(16, 7))
    fig6 = printClassificator(fig6, 121, x3, z3, dBayess, dMSE1, "MSE", ".")
    fig6 = printClassificator(fig6, 122, x3, y3, dBayess2, dMSE2, "MSE", "-")
    # fig6 = printClassificator(fig6, 122, x3, y3, dBayess2, dFisher2, "MSE")
    show()

    # посчитать экспериментальные ошибки классификаторов Фишера, СКО и Байесса

    # task 4.3
    # Классификатор Роббинса-Монро
    Z0 = np.ones((sizeX[0] + 2, sizeX[1]))
    Z0[-1] = Z0[-1]*-1
    Z1 = np.ones((sizeZ[0] + 2, sizeZ[1]))
    Z2 = np.ones((sizeY[0] + 2, sizeY[1]))

    Z0[0:sizeX[0], 0:sizeX[1]] = x3
    Z1[0:sizeZ[0], 0:sizeZ[1]] = z3
    Z2[0:sizeY[0], 0:sizeY[1]] = y3

    xz = []
    xy = []
    for i in range(sizeX[1]):
        xz.append(Z0[:, i])
        xz.append(Z1[:, i])

        xy.append(Z0[:, i])
        xy.append(Z2[:, i])

    xz = np.transpose(xz)
    xy = np.transpose(xy)

    # Z = np.concatenate((Z0, Z1), axis=1)

    Wrobbins1 = calcACRParameters(xz)
    arrBorders1 = [t3]
    for w in Wrobbins1:
        tmpY = borderLinClassificator(w[0:2], w[-1], t3, "Robbins-Monro with sample B")
        arrBorders1.append(tmpY[1])

    fig7 = plt.figure(figsize=(16, 7))
    fig7 = printClassificator(fig7, 121, x3, z3, arrBorders1[0:10], arrBorders1[0], "R", "--")
    resd1 = [arrBorders1[0], arrBorders1[-1]]
    fig7 = printClassificator(fig7, 122, x3, z3, resd1, dBayess, "R", "--")


    Wrobbins2 = calcACRParameters(xy)
    arrBorders2 = [t3]
    for w in Wrobbins2:
        tmpY = borderLinClassificator(w[0:2], w[-1], t3, "Robbins-Monro with sample B")
        arrBorders2.append(tmpY[1])

    fig8 = plt.figure(figsize=(16, 7))
    fig8 = printClassificator(fig8, 121, x3, y3, arrBorders2[0:10], arrBorders2[0], "R", "--")
    resd2 = [arrBorders2[0], arrBorders2[-1]]
    fig8 = printClassificator(fig8, 122, x3, y3, resd2, dBayess, "R", "--")

    # выбор начального W не влияет на скорость сходимости
    # последовательность: 0.5<b<=1 при большей степени коэфф 1/k^b становится меньше -> сходится плавнее, но медленней
    # и наоборот, при меньшей степени сходится быстрее, но может долго колебаться возле нужной гранницы

    show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
