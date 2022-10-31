import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imshow, show, imread
import math

M1 = [0, 0]
M2 = [1, 1]
M3 = [-1, 1]
M4 = [0, 1]
M5 = [0, 2]

B1 = [[0.05, 0.0], [0.0, 0.02]]
B2 = [[0.04, 0.01], [0.01, 0.05]]
B3 = [[0.02, 0.005], [0.005, 0.05]]
B4 = [[0.03, -0.01], [0.01, 0.03]]
B5 = [[0.04, 0.0], [0.0, 0.04]]

N = 50
n = 2


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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    A1 = calcMatrixA(B1)
    A2 = calcMatrixA(B2)
    A3 = calcMatrixA(B3)
    A4 = calcMatrixA(B4)
    A5 = calcMatrixA(B5)

    x1 = generate_vectors(A1, M1, n, N)
    x2 = generate_vectors(A2, M2, n, N)
    x3 = generate_vectors(A3, M3, n, N)
    x4 = generate_vectors(A4, M4, n, N)
    x5 = generate_vectors(A5, M5, n, N)

    data = np.concatenate((x1, x2, x3, x4, x5), axis=1)
    print(np.shape(data), data)

    fig1 = plt.figure(figsize=(10, 10))
    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.plot(x1[0], x1[1], 'r.')
    plt.plot(x2[0], x2[1], 'gx')
    plt.plot(x3[0], x3[1], 'b<')
    plt.plot(x4[0], x4[1], 'm*')
    plt.plot(x5[0], x5[1], 'c+')
    plt.legend(["class 1", "class 2", "class 3", "class 4", "class 5"])
    show()


