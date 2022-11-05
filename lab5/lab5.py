import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imshow, show, imread

M1 = [0, 0]
M2 = [1, 1]
M3 = [-1, 1]
M4 = [0, 1]
M5 = [0, 2]


B1 = [[0.03, 0.0], [0.0, 0.018]]
B2 = [[0.015, 0.0], [0.0, 0.03]]
B3 = [[0.02, 0.0], [0.0, 0.035]]
B4 = [[0.03, 0.0], [0.0, 0.03]]
B5 = [[0.03, 0.0], [0.0, 0.025]]
# B1 = [[0.05, 0.0], [0.0, 0.02]]
# B2 = [[0.04, 0.01], [0.01, 0.05]]
# B3 = [[0.02, 0.005], [0.005, 0.05]]
# B4 = [[0.03, -0.01], [0.01, 0.03]]
# B5 = [[0.04, 0.0], [0.0, 0.04]]

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


def d(x, z):
    dist = np.sum(np.square(x-z))
    return np.sqrt(dist)


Distance = np.vectorize(d, signature='(n),(m)->()')


def maxminMethod(vectors):
    result = copy.copy(vectors)
    clusters = []
    arrM = []
    M_all = vectors.sum(axis=1) / len(vectors[0])
    result = np.transpose(result)

    distances = Distance(result, M_all)
    m0 = result[np.argmax(distances)]
    clusters.append([m0])
    arrM.append(m0)
    result = np.delete(result, np.argmax(distances), axis=0)

    distances = Distance(result, m0)
    m1 = result[np.argmax(distances)]
    arrM.append(m1)
    clusters.append([m1])
    result = np.delete(result, np.argmax(distances), axis=0)

    dtypical = [Distance(m0, m1) / 2]
    dmin = [dtypical[-1] + 1]
    legends = ["M(x)", "class 0", "class 1"]

    # distanceTable
    #             x(0)          x(1)        ...       x(i)        ...       x(N-1)
    # M(0)      d(M0,x0)     d(M0, x1)      ...     d(M0, xi)     ...    d(M0, x(N-1))
    # M(1)      d(M1,x0)     d(M1, x1)      ...     d(M1, xi)     ...    d(M1, x(N-1))
    # ...         ...           ...         ...       ...         ...        ...
    # M(i)      d(Mi,x0)     d(Mi, x1)      ...     d(Mi, xi)     ...    d(Mi, x(N-1))
    # ...         ...           ...         ...       ...         ...        ...
    # M(L-2)  d(M(L-2),x0)  d(M(L-2), x1)   ...   d(M(L-2), xi)   ...   d(M(L-2), x(N-1))
    while dmin[-1] > dtypical[-1]:
        distanceTable = []
        for i in range(0, len(arrM)):
            distanceTable.append(Distance(result, arrM[i]))
        # распределить по существующим кластерам
        l = np.argmin(np.transpose(distanceTable), axis=1)
        tmp = copy.deepcopy(clusters)
        for k in range(0, len(result)):
            tmp[l[k]].append(result[k])

        # отобразить результат
        fig0 = plt.figure(figsize=(10, 10))
        viewClusters(tmp, arrM, fig0, 111, legend=legends)
        # show()

        # создание нового кластера(если надо)
        minDistances = np.min(np.transpose(distanceTable), axis=1)
        M_ = result[np.argmax(minDistances)]
        dmin.append(np.min(Distance(arrM, M_)))
        if dmin[-1] > dtypical[-1]:
            legends.append(f"class {len(arrM)}")
            arrM.append(M_)
            clusters.append([M_])
            result = np.delete(result, np.argmax(minDistances), axis=0)
            dtypical.append(0)
            for j in range(0, len(arrM)):
                dtypical[-1] += np.sum(Distance(arrM, arrM[j]))
            dtypical[-1] /= 2*len(arrM)*(len(arrM) - 1)
    dmin.pop(0)
    for k in range(0, len(result)):
        clusters[l[k]].append(result[k])
    return clusters, dmin, dtypical, arrM


def K_introGroupAvg(vectors, initVectors):
    K = len(np.transpose(initVectors))
    clusters = []
    legends = ["M(x)"]
    for i in range(0, K):
        clusters.append([])
        legends.append(f"class {i}")

    result = np.transpose(vectors)
    new_arrM = copy.deepcopy(np.transpose(initVectors))
    prev_arrM = np.mean(result, axis=0)*np.ones_like(new_arrM)
    prev_k = np.zeros((len(result),)).astype(int)
    imposters = [-1]

    tmp = [result]
    while not (imposters[-1] == 0):  # not ((imposters[-1] == 0) | (new_arrM == prev_arrM).all())
        distances = []
        for i in range(0, K):
            distances.append(Distance(result, new_arrM[i]))
        # принадлежность классам
        new_k = np.argmin(distances, axis=0)

        # отобразить предыдущее разбиение
        fig0 = plt.figure(figsize=(16, 7))
        viewClusters(tmp, prev_arrM, fig0, 121, legends)

        # разбиение по классам
        copyClusters = copy.deepcopy(clusters)
        for i in range(0, len(result)):
            copyClusters[new_k[i]].append(result[i])

        # отобразить текущее разбиение
        tmp = copy.copy(copyClusters)
        viewClusters(copyClusters, new_arrM, fig0, 122, legends)

        # количество изменивших класс
        values, counts = np.unique(new_k == prev_k, return_counts=True)
        if False in list(values):
            imposters.append(counts[list(values).index(False)])
        else:
            imposters.append(0)
        prev_k = copy.copy(new_k)

        # обновление мат.ожиданий
        prev_arrM = copy.copy(new_arrM)
        for i in range(0, K):
            new_arrM[i] = np.mean(copyClusters[i], axis=0)
    imposters.pop(0)
    for j in range(0, len(result)):
        clusters[new_k[j]].append(result[j])
    return clusters, new_arrM, imposters


def viewClusters(datas, arrM, fig, loc, legend):
    viewData = []
    for k in range(0, len(datas)):
        viewData.append(np.transpose(datas[k]))
    tmp = np.transpose(arrM)

    fig.add_subplot(loc)
    plt.xlim(-1.6, 1.6)
    plt.ylim(-0.6, 2.6)
    plt.plot(tmp[0], tmp[1], 'ko')
    c = ['r.', 'b.', 'g.', 'c.', 'm.', 'y.']
    for i in range(0, len(viewData)):
        plt.plot(viewData[i][0], viewData[i][1], c[i % len(c)])
    plt.legend(legend)
    return fig


def viewResultAndOriginal(classes, Ms, legends):
    fig = plt.figure(figsize=(16, 7))
    fig.add_subplot(121)
    plt.xlim(-1.6, 1.6)
    plt.ylim(-0.6, 2.6)
    plt.plot(x1[0], x1[1], 'r.')
    plt.plot(x2[0], x2[1], 'gx')
    plt.plot(x3[0], x3[1], 'b<')
    plt.plot(x4[0], x4[1], 'm*')
    plt.plot(x5[0], x5[1], 'c+')
    plt.legend(legs)
    viewClusters(classes, Ms, fig, 122, legends)


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

    classes, d_min, d_typical, arr_M = maxminMethod(data)

    legs = ["M(x)"]
    for i in range(0, len(classes)):
        legs.append(f"class {i}")
    viewResultAndOriginal(classes, arr_M, legs)

    x = np.arange(2, 2+len(d_min), 1)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, d_min, c="b", marker="o", linestyle="-")
    plt.plot(x, d_typical, c="orange", marker="o", linestyle="-")
    plt.legend(["d min", "d typical"])
    show()

    classes3, M3, imposter3 = K_introGroupAvg(data, data[:, [151, 156, 160]])
    show()

    classes5, M5, imposters5 = K_introGroupAvg(data, data[:, [151, 156, 160, 161, 166]])
    viewResultAndOriginal(classes5, M5, legs)
    show()

    fakeClasses5, fakeM5, fakeImposters5 = K_introGroupAvg(data, data[:, 0:5])  # подумать над начальными
                                                                                # для неправильной
    viewResultAndOriginal(fakeClasses5, fakeM5, legs)
    show()


