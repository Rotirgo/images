import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show
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
B3 = [[0.25, -0.05],
      [-0.05, 0.25]]


def calculate_P_matrix(dataset, r, kernel_func):
    size = dataset.shape
    result = np.zeros((size[1], size[1]))
    for i in range(0, size[1]):
        for j in range(0, size[1]):
            result[i, j] = r[i]*r[j]*kernel_func(dataset[:, i], dataset[:, j])
    return result


def calcW(x, r, l):
    if l is None:
        print("Bad")
        return [1, 1], 0
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
    fig.add_subplot(pos)
    plt.title(f"{name}")
    xlim = plt.xlim(-2, 3)
    ylim = plt.ylim(-2, 3)
    plt.plot(classes[0][0], classes[0][1], 'r+', label="class 0")
    plt.plot(classes[1][0], classes[1][1], 'bx', label="class 1")
    plt.plot(borders[0][0], borders[0][1], 'm-', label=borderNames[0], alpha=0.5)

    # ширина полосы при квадратичном программировании
    plt.plot(borders[1][0], borders[1][1], 'm--', label=borderNames[1], alpha=0.3)
    plt.plot(borders[2][0], borders[2][1], 'm--', alpha=0.3)
    # legend1 = plt.legend(loc=1)

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z0 = SVC.decision_function(xy).reshape(XX.shape)

    # plot support vectors
    if isinstance(SVC, type(svm.SVC())):
        plt.scatter(SVC.support_vectors_[:, 0], SVC.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
                    edgecolors='k', label="Support Vectors")
    legend1 = plt.legend(loc=1)

    # plot decision boundary and margins
    CS = plt.contour(XX, YY, Z0, colors='green', levels=[-1, 0, 1], alpha=0.3, linestyles=['--', '-', '--'])
    artists, labels = CS.legend_elements()
    custom_labels = []
    for level, contour in zip([-1, 0, 1], CS.collections):
        custom_labels.append(f'{SVM_labels[level%2]}')
    plt.legend(artists[0:-1], custom_labels[0:-1], loc="upper left")
    plt.gca().add_artist(legend1)

    return fig


def analiseSVMkernels(Cs, X, y, kernParam, classes, borders, kernelname):
    for i in range(0, len(Cs)):
        svc_kernel = svm.SVC(kernel=kernParam["kernel"], gamma=kernParam["gamma"],
                             coef0=kernParam["coef0"], degree=kernParam["degree"], C=Cs[i])
        svc_kernel.fit(X=X, y=y)
        fig3 = plt.figure(figsize=(7, 7))
        bords = [borders[0][i], borders[1][i], borders[2][i]]
        fig3 = viewFig(fig3, classes, bords, 111, f"SVC {kernelname} with C:{Cs[i]}",
                       ["SVM quadprog", "SVM qp range"], svc_kernel, [f"SVC {kernelname} range", f"SVC {kernelname}"])
    show()


def border_and_range(w, Wn):
    t = np.linspace(-5, 5, 100)
    border = lab1.borderLinClassificator(w, Wn, t, "SVM")
    border_up = lab1.borderLinClassificator(w, (Wn + 1 / np.dot(w, w)), t, "SVM")
    border_low = lab1.borderLinClassificator(w, (Wn - 1 / np.dot(w, w)), t, "SVM")
    return border, border_up, border_low


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')

    # task 1
    A1 = lab1.calcMatrixA(B1)
    A2 = lab1.calcMatrixA(B2)
    A3 = lab1.calcMatrixA(B3)

    x = lab1.generate_vectors(A1, M1, n, N)
    y = lab1.generate_vectors(A2, M2, n, N)
    z = lab1.generate_vectors(A3, M2, n, N)

    vector_r = np.ones(2*N)
    vector_r[0:N] *= -1
    datasetXY = np.concatenate([x, y], axis=1)
    datasetXZ = np.concatenate([x, z], axis=1)

    # task 2
    # линейно разделимые
    Pxy = calculate_P_matrix(datasetXY, vector_r, kernel_func=np.dot)
    q = -np.ones(2 * N)
    G = -np.eye(2 * N)
    h = np.zeros(2 * N)
    A = csc_matrix(vector_r)
    b = [0.]

    # не всегда находит решение из-за ограничения в количестве итераций
    # запускать несколько раз
    ls = osqp_solve_qp(P=csc_matrix(Pxy), q=q, G=G, h=h, A=csc_matrix(A), b=b, max_iter=100000)
    print(np.shape(ls))
    W, wn = calcW(datasetXY, vector_r, ls)
    if ls is not None:
        print(f"important ls: {len(ls[ls > 1e-3])}\t{ls[ls > 1e-3]}")
    t = np.linspace(-5, 5, 100)
    border_qp = border_and_range(W, wn)

    yTrain = np.zeros(2*N)
    yTrain[N:2*N] = np.ones(N)
    xTrain = datasetXY.T
    svc = svm.SVC(kernel='linear', C=1)
    lin_svc = svm.LinearSVC(dual=True, C=1)
    svc.fit(X=xTrain, y=yTrain)  # libsvm
    lin_svc.fit(X=xTrain, y=yTrain)  # liblinear

    fig1 = plt.figure(figsize=(16, 7))
    fig1 = viewFig(fig1, [x, y], border_qp, 121, "SVC borders",
                   ["SVM quadprog", "SVM qp range"], svc, ["SVC range", "SVC"])
    fig1 = viewFig(fig1, [x, y], border_qp, 122, "Linear SVC borders",
                   ["SVM quadprog", "SVM qp range"], lin_svc, ["lin SVC range", "lin SVC"])

    # task 3
    my_C = 20
    C = [0.1, 1, 10, my_C]
    Pxz = calculate_P_matrix(datasetXZ, vector_r, kernel_func=np.dot)
    G_withC = np.concatenate((G, np.eye(2 * N)), axis=0)

    limbs = []
    for i in range(0, len(C)):
        h_withC = np.concatenate((h, C[i] * np.ones(2 * N)))
        limbs.append(osqp_solve_qp(P=csc_matrix(Pxz), q=q, G=G_withC, h=h_withC, A=csc_matrix(A), b=b, max_iter=50000))
    print("Good")

    borderXZ = []
    borderXZ_up = []
    borderXZ_low = []
    for i in range(0, len(C)):
        W2, wn2 = calcW(datasetXZ, vector_r, limbs[i])
        if ls is not None:
            print(f"important ls: {len(limbs[i][limbs[i] > 1e-3])}")
        border_qp = border_and_range(W2, wn2)
        borderXZ.append(border_qp[0])
        borderXZ_up.append(border_qp[1])
        borderXZ_low.append(border_qp[2])

        xTrain = datasetXZ.T
        svc2 = svm.SVC(kernel='linear', C=C[i])
        svc2.fit(X=xTrain, y=yTrain)  # libsvm

        if i % 2 == 0:
            fig2 = plt.figure(figsize=(16, 7))
        fig2 = viewFig(fig2, [x, z], border_qp, 121+(i % 2), f"SVC borders with C:{C[i]}",
                       ["SVM quadprog", "SVM qp range"], svc2, ["SVC range", "SVC"])
    show()

    # task 4
    # kernel, gamma, coef0, degree
    dict_params = {"kernel": "poly", "gamma": "scale", "coef0": 0.0, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z],
                      [borderXZ, borderXZ_up, borderXZ_low], "polynomial")

    dict_params = {"kernel": "poly", "gamma": "scale", "coef0": 1.0, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z],
                      [borderXZ, borderXZ_up, borderXZ_low], "polynomial not simple")

    # "scale" = 1/(n_features * X.var()) , "auto" = 1/n_features
    dict_params = {"kernel": "rbf", "gamma": "scale", "coef0": 0.0, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z],
                      [borderXZ, borderXZ_up, borderXZ_low], "radiance func")

    D = 2*xTrain.var()
    dict_params = {"kernel": "rbf", "gamma": 1 / D, "coef0": 0.0, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z],
                      [borderXZ, borderXZ_up, borderXZ_low], "radiance func Gauss")

    dict_params = {"kernel": "sigmoid", "gamma": 0.5, "coef0": -0.01, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z],
                      [borderXZ, borderXZ_up, borderXZ_low], "sigmoid")

    print("Wow! It is work!")
