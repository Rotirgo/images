import numpy as np
import qpsolvers
from matplotlib import pyplot as plt
from skimage.io import show
import lab5.lab5 as lab5
import lab1_2_4.lab124 as lab1
from scipy.sparse import csc_matrix
from qpsolvers.solvers.osqp_ import osqp_solve_qp
from sklearn import svm

N = 100
n = 2

M1 = [0, 0]
M2 = [1, 1]

B1 = [[0.05, -0.0],
      [-0.0, 0.05]]
B2 = [[0.2, -0.18],
      [-0.18, 0.2]]
B3 = [[0.25, -0.07],
      [-0.07, 0.25]]


def calculate_P_matrix(dataset, r):
    size = dataset.shape
    result = np.zeros((size[1], size[1]))
    for i in range(0, size[1]):
        for j in range(0, size[1]):
            result[i, j] = r[i]*r[j]*np.dot(dataset[:, i], dataset[:, j])
    return result


def calcW(x, r, l):
    w = 0
    for i in range(0, len(r)):
        w += r[i]*l[i]*x[:, i]

    J = l[l > 1e-3]
    wn = 0
    for i in range(0, len(J)):
        idx = list(l).index(J[i])
        wn += r[idx] - np.dot(w, x[:, idx])
    wn /= len(J)
    return w, wn


def viewFig(fig, classes, borders, pos, name, borderNames, SVC, SVM_labels):
    fig1.add_subplot(pos)
    plt.title(f"{name}")
    xlim = plt.xlim(-2, 3)
    ylim = plt.ylim(-2, 3)
    plt.plot(classes[0][0], classes[0][1], 'r+', label="class 0")
    plt.plot(classes[1][0], classes[1][1], 'bx', label="class 1")
    plt.plot(borders[0][0], borders[0][1], 'm--', label=borderNames[0])
    legend1 = plt.legend(loc=1)

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z0 = SVC.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    CS = plt.contour(XX, YY, Z0, colors='green', levels=[-1, 0, 1], alpha=0.3, linestyles=['--', '-', '--'])
    artists, labels = CS.legend_elements()
    custom_labels = []
    for level, contour in zip([-1, 0, 1], CS.collections):
        custom_labels.append(f'{SVM_labels[level%2]}')
    plt.legend(artists[0:-1], custom_labels[0:-1], loc="upper left")
    plt.gca().add_artist(legend1)

    # plot support vectors
    if isinstance(SVC, type(svm.SVC())):
        plt.scatter(SVC.support_vectors_[:, 0], SVC.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
                    edgecolors='k', label="Support Vectors")
    return fig


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')

    A1 = lab1.calcMatrixA(B1)
    A2 = lab1.calcMatrixA(B2)
    A3 = lab1.calcMatrixA(B3)

    x = lab1.generate_vectors(A1, M1, n, N)
    y = lab1.generate_vectors(A2, M2, n, N)
    z = lab1.generate_vectors(A3, M2, n, N)

    vector_r = np.ones(2*N)
    vector_r[0:N] *= -1
    datasetXY = np.concatenate([x, y], axis=1)
    datasetXZ = np.concatenate([x, y], axis=1)

    # линейно разделимые
    Pxy = calculate_P_matrix(datasetXY, vector_r)
    Pxz = calculate_P_matrix(datasetXZ, vector_r)
    q = -np.ones(2 * N)
    G = -np.eye(2 * N)
    h = np.zeros(2 * N)
    A = csc_matrix(vector_r)
    b = [0.]

    # не всегда находит решение из-за ограничения в количестве итераций
    # запускать несколько раз
    ls = osqp_solve_qp(P=csc_matrix(Pxy), q=q, G=G, h=h, A=csc_matrix(A), b=b, max_iter=50000)
    limbs = osqp_solve_qp(P=csc_matrix(Pxz), q=q, G=G, h=h, A=csc_matrix(A), b=b, max_iter=50000)

    W, wn = calcW(datasetXY, vector_r, ls)
    print(f"important ls: {len(ls[ls > 1e-3])}\t{ls[ls > 1e-3]}")
    t = np.linspace(-5, 5, 100)
    borderXY = lab1.borderLinClassificator(W, wn, t, "SVM")

    yTrain = np.zeros(2*N)
    yTrain[N:2*N] = np.ones(N)
    xTrain = datasetXY.T
    svc = svm.SVC(kernel='linear', C=1)
    lin_svc = svm.LinearSVC(dual=True, C=1)
    svc.fit(X=xTrain, y=yTrain)  # libsvm
    lin_svc.fit(X=xTrain, y=yTrain)  # liblinear

    fig1 = plt.figure(figsize=(16, 7))
    fig1 = viewFig(fig1, [x, y], [borderXY], 121, "SVC borders",
                   ["SVM"], svc, ["SVC range", "SVC"])
    fig1 = viewFig(fig1, [x, y], [borderXY], 122, "Linear SVC borders",
                   ["lin SVM"], lin_svc, ["lin SVC range", "lin SVC"])
    show()


