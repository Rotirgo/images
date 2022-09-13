# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imshow, show, imread
from scipy.special import erf, erfinv
import math
from scipy.interpolate import interp1d

var = 1
M1 = [0, 0]
M2 = [1, 1]
M3 = [-1, 1]
B1 = [[0.12, 0.0],
      [0.0, 0.23]]
B2 = [[0.23, 0.0],
      [0.0, 0.17]]
B3 = [[0.2, 0.1],
      [0.1, 0.3]]

n = 2
N = 200

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
     [0, 0, 0, 0, 1, 0, 0, 0, 0], ]


def generate_vectors(A, M, n, N):
    left_border = 0
    right_border = 1
    m = (right_border + left_border) / 2
    k = 20
    Sn = np.zeros((n, N))
    for i in range(0, k, 1):
        Sn += np.random.uniform(left_border, right_border, (n, N)) - m
    CKO = (right_border - left_border) / np.sqrt(12)
    E = Sn / (CKO * np.sqrt(k))
    x = np.matmul(A, E) + np.reshape(M, (2, 1)) * np.ones((1, N))
    return x


def calcMatA(B):
    A = np.zeros((2, 2))
    A[0][0] = np.sqrt(B[0][0])
    A[0][1] = 0
    A[1][0] = B[0][1] / np.sqrt(B[0][0])
    A[1][1] = np.sqrt(B[1][1] - (B[0][1] ** 2) / B[0][0])
    return A


def M(x):
    M = np.sum(x, axis=1) / N
    return M


def B(x, M):
    B = np.zeros((2, 2))
    for i in range(0, N, 1):
        tmp = np.reshape(x[:, i], (2, 1))
        B += (np.matmul(tmp, np.transpose(tmp)))
    B /= N
    B -= np.matmul(np.reshape(M, (2, 1)), np.transpose(np.reshape(M, (2, 1))))
    return B

#lab 2
def BayesBorderSampleB(m1, m2, b):
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    sumM = np.reshape(m1, (1, 2)) + np.reshape(m2, (1, 2))
    tmp1 = np.matmul(sumM, np.linalg.inv(b))
    tmp2 = 0.5 * np.reshape(np.matmul(tmp1, np.reshape(difM, (2, 1))), (1, ))
    tmp3 = np.reshape(np.matmul(difM, np.linalg.inv(b)), (2, ))
    k = -tmp3[0]/tmp3[1]
    c = tmp2[0]/tmp3[1]
    x = np.arange(-3, 3, 0.1)
    y = k*x + c
    return x, y


def Phi(x):
    return 0.5*(1 + erf(x/np.sqrt(2)))


def invPhi(x):
    return np.sqrt(2)*erfinv(2*x-1)


def p_error(m1, m2, b):
    difM = np.reshape(m1, (1, 2)) - np.reshape(m2, (1, 2))
    MahalnobisDistant = np.reshape(np.matmul(np.matmul(difM, b), np.transpose(difM)), (1,))
    p = [0, 0]
    p[0] = 1 - Phi(0.5*np.sqrt(MahalnobisDistant[0]))
    p[1] = Phi(-0.5*np.sqrt(MahalnobisDistant[0]))
    return p


def BayesBorderDifferenceB(m1, m2, b1, b2):
    divB = math.log(np.linalg.det(b1)/np.linalg.det(b2))
    dist1 = np.matmul(np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b1)), np.reshape(m1, (2, 1)))
    dist2 = np.matmul(np.matmul(np.reshape(m2, (1, 2)), np.linalg.inv(b2)), np.reshape(m2, (2, 1)))
    a = np.linalg.inv(b2) - np.linalg.inv(b1)
    b = np.reshape(2*(np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b1))-np.matmul(np.reshape(m2, (1, 2)),
                                                                                     np.linalg.inv(b2))), (2,))
    c = np.reshape((divB - dist1 + dist2), (1,))
    x = np.arange(-3, 3, 0.00001)
    D = (np.square((a[1][0] + a[0][1]))-4*a[0][0]*a[1][1])*np.square(x) + \
        (2*(a[1][0] + a[0][1])*b[1] - 4*b[0]*a[1][1])*x + b[1]*b[1] - 4*a[1][1]*c[0]
    #print(D)
    y1 = (-(a[1][0] + a[0][1]) * x - b[1] + np.sqrt(D)) / (2*a[1][1])
    y2 = (-(a[1][0] + a[0][1]) * x - b[1] - np.sqrt(D)) / (2*a[1][1])
    return x, y1, y2


def classificatorMinMax():
    return 0


def borderMinMax():
    return 0


def classificatorNP(x, p0, m1, m2, b, className1, className2):
    tmp0 = np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b))
    distMahal = np.matmul(tmp0, np.reshape(m2, (2, 1)))
    ro = distMahal[0][0]
    lymbda = -0.5 * ro + np.sqrt(ro) * invPhi(1 - p0)
    l = np.exp(lymbda)
    tmp1 = np.matmul(np.reshape(x, (2, 1)) - np.reshape(m1, (1, 2)), np.linalg.inv(b))
    r1 = np.matmul(tmp1, np.reshape(x, (2, 1)) - np.reshape(m1, (2, 1)))
    tmp2 = np.matmul(np.reshape(x, (2, 1)) - np.reshape(m2, (1, 2)), np.linalg.inv(b))
    r2 = np.matmul(tmp2, np.reshape(x, (2, 1)) - np.reshape(m2, (2, 1)))
    y = np.exp(-0.5*(r2[0][0]-r1[0][0]))
    if (y > l): return className2
    return className1


def borderNPclass(p0, m1, m2, b):
    tmp1 = np.matmul(np.reshape(m1, (1, 2)), np.linalg.inv(b))
    distMahal = np.matmul(tmp1, np.reshape(m2, (2, 1)))
    ro = distMahal[0][0]
    lymbda = -0.5*ro + np.sqrt(ro)*invPhi(1-p0)
    difM = np.reshape(m1, (2, 1)) - np.reshape(m2, (2, 1))
    sumM = np.reshape(m1, (2, 1)) + np.reshape(m2, (2, 1))
    a = np.matmul(np.linalg.inv(b), difM)
    d = np.matmul(np.transpose(difM), np.linalg.inv(b))
    tmp2 = np.matmul(np.transpose(difM), np.linalg.inv(b))
    c = -np.matmul(tmp2, sumM)
    k = -(a[0][0]+d[0][0])/(a[1][0]+d[0][1])
    l = -(c[0][0]+2*lymbda)/(a[1][0]+d[0][1])
    x = np.arange(-3, 3, 0.1)
    y = k * x + l
    return x, y


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # task 1.2
    A = calcMatA(B1)

    x1 = generate_vectors(A, M1, n, N)
    y1 = generate_vectors(A, M2, n, N)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(x1[0], x1[1], 'r.')
    plt.plot(y1[0], y1[1], 'b*')
    show()

    # task 1.3
    A1 = calcMatA(B1)
    A2 = calcMatA(B2)
    A3 = calcMatA(B3)

    x2 = generate_vectors(A1, M1, n, N)
    y2 = generate_vectors(A2, M2, n, N)
    z2 = generate_vectors(A3, M3, n, N)

    fig1 = plt.figure(figsize=(10, 10))
    plt.xlim(-3, 3)
    plt.ylim(-2, 4)
    plt.plot(x2[0], x2[1], 'r.')
    plt.plot(y2[0], y2[1], 'gx')
    plt.plot(z2[0], z2[1], 'b+')
    show()

    # task 1.4
    Mx1 = M(x1)
    My1 = M(y1)
    Mx2 = M(x2)
    My2 = M(y2)
    Mz2 = M(z2)

    Bx1 = B(x1, Mx1)
    By1 = B(y1, My1)
    Bx2 = B(x2, Mx2)
    By2 = B(y2, My2)
    Bz2 = B(z2, Mz2)

    print(f'M1: {M1} <-- [{Mx1[0]:.3f}, {Mx1[1]:.3f}]')
    print(f"M1: {M1} <-- [{My1[0]:.3f}, {My1[1]:.3f}]")
    print(f"M1: {M1} <-- [{Mx2[0]:.3f}, {Mx2[1]:.3f}]")
    print(f"M2: {M2} <-- [{My2[0]:.3f}, {My2[1]:.3f}]")
    print(f"M3: {M3} <-- [{Mz2[0]:.3f}, {Mz2[1]:.3f}]\n")
    print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{Bx1[0][0]:.3f}, {Bx1[0][1]:.3f}],")
    print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{Bx1[1][0]:.3f}, {Bx1[1][1]:.3f}]\n")
    print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{By1[0][0]:.3f}, {By1[0][1]:.3f}],")
    print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{By1[1][0]:.3f}, {By1[1][1]:.3f}]\n")
    print(f"B1: [{B1[0][0]:.2f}, {B1[0][1]:.2f}],   <--  [{Bx2[0][0]:.3f}, {Bx2[0][1]:.3f}],")
    print(f"    [{B1[1][0]:.2f}, {B1[1][1]:.2f}]         [{Bx2[1][0]:.3f}, {Bx2[1][1]:.3f}]\n")
    print(f"B2: [{B2[0][0]:.2f}, {B2[0][1]:.2f}],   <--  [{By2[0][0]:.3f}, {By2[0][1]:.3f}],")
    print(f"    [{B2[1][0]:.2f}, {B2[1][1]:.2f}]         [{By2[1][0]:.3f}, {By2[1][1]:.3f}]\n")
    print(f"B3: [{B3[0][0]:.2f}, {B3[0][1]:.2f}],   <--  [{Bz2[0][0]:.3f}, {Bz2[0][1]:.3f}],")
    print(f"    [{B3[1][0]:.2f}, {B3[1][1]:.2f}]         [{Bz2[1][0]:.3f}, {Bz2[1][1]:.3f}]")

    # task 1.5
    # ???

    # task 2.1
    # x1 и y1 - байессовская граница
    d1 = BayesBorderSampleB(M1, M2, B1)
    p = p_error(M1, M2, B1)
    R = p[0]
    print(p)


    # task 2.2
    # x1 и y1 - классификаторы минимаксный и Неймана-Пирсона
    # минимаксный спросить

    # Неймана-Пирсона
    # (при нормальном распределении с одинаковыми B получим байессовский классификатор с простейшей матрицей c_ij)
    # поэтому границы раздела совпадают
    p0 = 0.05
    d3 = borderNPclass(p0, M1, M2, B1)
    x = [0.5, 0.5]
    result = classificatorNP(x, p0, M1, M2, B1, "red", "blue")
    print(result)

    fig2 = plt.figure(figsize=(10, 10))
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.plot(x1[0], x1[1], 'r.')
    plt.plot(y1[0], y1[1], 'b*')
    plt.plot(d1[0], d1[1], 'y-')
    plt.plot(d3[0], d3[1], 'm--')
    plt.plot(x[0], x[1], 'ko')
    show()


    # task 2.3
    # x2, y2 и z2 - байессовская граница
    dxy = BayesBorderDifferenceB(M1, M2, B1, B2)
    dxz = BayesBorderDifferenceB(M1, M3, B1, B3)
    dyz = BayesBorderDifferenceB(M2, M3, B2, B3)
    fig4 = plt.figure(figsize=(10, 10))
    plt.xlim(-3, 3)
    plt.ylim(-2, 4)
    plt.plot(x2[0], x2[1], 'r.')
    plt.plot(y2[0], y2[1], 'gx')
    plt.plot(z2[0], z2[1], 'b+')
    plt.plot(dxy[0], dxy[1], 'y-') # в экран не влазит, можно не показывать
    plt.plot(dxy[0], dxy[2], 'y-')
    plt.plot(dxz[0], dxz[1], 'c-')
    plt.plot(dxz[0], dxz[2], 'c-')
    plt.plot(dyz[0], dyz[1], 'm-')
    plt.plot(dyz[0], dyz[2], 'm-')
    show()

    # для двух классов определить экспериментально вероятности ошибочной классификации
    # определить размер N чтобы погрешность была 5%

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
