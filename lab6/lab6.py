import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show
import lab1_2_4.lab124 as lab1
from scipy.sparse import csc_matrix
from qpsolvers.solvers.osqp_ import osqp_solve_qp
from qpsolvers.solvers.cvxopt_ import cvxopt_solve_qp  # попробовать с эти qp
from sklearn import svm

N = 100
n = 2
eps = 1e-5

M1 = [0, 0]
M2 = [1, 1]

B1 = [[0.05, -0.0],
      [-0.0, 0.05]]
B2 = [[0.2, -0.18],
      [-0.18, 0.2]]
B3 = [[0.25, -0.05],
      [-0.05, 0.25]]


def calculate_P_matrix(dataset, r, kernel_func, **kwargs):
    size = dataset.shape
    result = np.zeros((size[1], size[1]))
    for i in range(0, size[1]):
        for j in range(0, size[1]):
            if len(kwargs) < 1:
                result[i, j] = r[i]*r[j]*kernel_func(dataset[:, i], dataset[:, j])
            else:
                result[i, j] = r[i] * r[j] * kernel_func(x=dataset[:, i], y=dataset[:, j], p=kwargs)
    return result


def calcW(x, r, l):
    if l is None:
        print("Bad")
        return [1, 1], 0
    w = 0
    for i in range(0, len(r)):
        w += r[i]*l[i]*x[:, i]

    J = l[l > eps]
    wn = 0
    for i in range(0, len(J)):
        idx = list(l).index(J[i])
        wn += r[idx] - np.dot(w, x[:, idx])
    wn /= len(J)
    return w, wn


def calcW_not_lin(x, r, l, **kwargs):
    supps = np.transpose(kwargs["supX"])
    wn = 0
    cnt = -1
    for i in range(0, len(l)):
        if l[i] > eps:
            cnt += 1
            w_x = 0
            for j in range(0, len(r)):
                w_x += l[j] * r[j] * kwargs["kernel"](x=x[:, j], y=supps[cnt], p=kwargs["p"])
            wn += r[i] - w_x
    wn /= (cnt + 1)
    return wn


def getSupportVectors(lymbs, dataset):
    # если смотреть еще очень маленькие лямбды(1e-21), то опорных векторов много
    # и они в основном за пределами разделительной полосы
    # возможно это из-за G*l<h
    tmp = lymbs[lymbs > eps]  # оставил только сильно значимые лямды
    sup_vectors = []
    for el in tmp:
        sup_vectors.append(dataset[:, list(lymbs).index(el)])
    return np.transpose(sup_vectors)


def getContourLabel(contour, names, levels, loc):
    artists, labels = contour.legend_elements()
    custom_labels = []
    for level, contour in zip(levels, contour.collections):
        custom_labels.append(f'{names[level % 2]}')
    if len(names) > 1:
        legend = plt.legend(artists[0:-1], custom_labels[0:-1], loc=loc)
    else:
        legend = plt.legend(artists, custom_labels, loc=loc)
    return legend


def viewFig(fig, classes, pos, name, borderNames, SVC, SVM_labels, qp_supVectors, Type, limbs, **kwargs):
    fig.add_subplot(pos)
    plt.title(f"{name}")
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.plot(classes[0][0], classes[0][1], 'r+', label="class 0")
    plt.plot(classes[1][0], classes[1][1], 'bx', label="class 1")

    # create grid to evaluate model
    xx = np.linspace(-2, 3, 50)
    yy = np.linspace(-2, 3, 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z0 = SVC.decision_function(xy).reshape(XX.shape)

    # расчитать границу для ручного svm
    dataset = np.concatenate(classes, axis=1)
    arr_r = np.ones(2 * N)
    arr_r[0:N] *= -1
    if Type == "lin":
        W, wn = calcW(dataset, arr_r, limbs)
        Wx = np.matmul(W, xy.T)
        ZZ = (Wx + wn).reshape(XX.shape)
        hand = plt.contour(XX, YY, ZZ, colors='m', levels=[-1, 0, 1], alpha=0.3, linestyles=['--', '-', '--'])
        legend2 = getContourLabel(hand, borderNames, [-1, 0, 1], 3)

        cntErrs = 0
        for i in range(0, len(arr_r)):
            zz = np.matmul(W, dataset[:, i]) + wn
            if ((zz < 0) & (arr_r[i] == 1)) | ((zz > 0) & (arr_r[i] == -1)):
                cntErrs += 1
        print(f"{name}\thas {cntErrs} errors classification ({100 * cntErrs / len(arr_r):.2f}%)")
    else:
        kernelFunc = kwargs["params"]["kernel"]
        ZZ = np.zeros(len(xy))
        for i in range(0, len(xy)):
            for j in range(0, len(arr_r)):
                ZZ[i] += limbs[j]*arr_r[j]*kernelFunc(dataset[:, j], xy[i], p=kwargs["params"])
        wn = calcW_not_lin(dataset, arr_r, limbs, supX=qp_supVectors,
                           kernel=kwargs["params"]["kernel"], p=kwargs["params"])
        ZZ = ZZ.reshape(XX.shape) + wn
        hand = plt.contour(XX, YY, ZZ, colors='m', levels=[0], alpha=0.3, linestyles=['-'])
        legend2 = getContourLabel(hand, [borderNames[0]], [0], 3)

        # классы здесь есть, массив r есть, wn есть
        # функция для классификации
        cntErrs = 0
        for i in range(0, len(arr_r)):
            zz = 0
            for j in range(0, len(arr_r)):
                zz += limbs[j] * arr_r[j] * kernelFunc(dataset[:, j], dataset[:, i], p=kwargs["params"])
            zz += wn
            # print(arr_r[i], zz)
            if ((zz < 0)&(arr_r[i] == 1))|((zz > 0)&(arr_r[i] == -1)):
                cntErrs += 1
        print(f"{name} has {cntErrs} errors classification ({100*cntErrs/len(arr_r):.2f}%)")


    # plot support vectors
    if isinstance(SVC, type(svm.SVC())):
        plt.scatter(SVC.support_vectors_[:, 0], SVC.support_vectors_[:, 1], s=90, linewidth=1, facecolors='none',
                    edgecolors='k', label="Support Vectors SVC", alpha=0.7)
    plt.scatter(qp_supVectors[0], qp_supVectors[1], s=130, linewidth=1, facecolors='none',
                edgecolors='orange', label="Support Vectors qp", alpha=0.5)
    legend1 = plt.legend(loc=1)

    # plot decision boundary and margins
    CS = plt.contour(XX, YY, Z0, colors='green', levels=[-1, 0, 1], alpha=0.3,
                     linestyles=['--', '-', '--'])
    artists, labels = CS.legend_elements()
    custom_labels = []
    for level, contour in zip([-1, 0, 1], CS.collections):
        custom_labels.append(f'{SVM_labels[level%2]}')
    plt.legend(artists[0:-1], custom_labels[0:-1], loc="upper left")
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    return fig


def analiseSVMkernels(Cs, X, y, kernParam, classes, kernelname, qp_supVectors,
                      Type, limbs, **kwargs):
    for i in range(len(Cs) - 1, -1, -1):
        svc_kernel = svm.SVC(kernel=kernParam["kernel"], gamma=kernParam["gamma"],
                             coef0=kernParam["coef0"], degree=kernParam["degree"], C=Cs[i])
        svc_kernel.fit(X=X, y=y)
        fig3 = plt.figure(figsize=(7, 7))
        fig3 = viewFig(fig3, classes, 111, f"SVC {kernelname} with C:{Cs[i]}",
                       ["SVM quadprog", "SVM qp range"], svc_kernel,
                       [f"SVC {kernelname} range", f"SVC {kernelname}"], qp_supVectors[i],
                       Type, limbs[i], params=kwargs)
    print("\n")
    show()


def K_poly0(x, y, **kwargs):
    degree = kwargs["p"]["d"]
    K = np.dot(x, y)**degree
    return K

def K_poly1(x, y, **kwargs):
    degree = kwargs["p"]["d"]
    K = (np.dot(x, y) + 1)**degree
    return K

def K_rad(x, y, **kwargs):
    gamma = kwargs["p"]["gamma"]
    dif = x - y
    K = np.exp(-gamma * np.dot(dif, dif))
    return K

def K_radGauss(x, y, **kwargs):
    D = kwargs["p"]["gamma"]
    dif = x - y
    K = np.exp(-D * np.dot(dif, dif))
    return K

def K_sigmoid(x, y, **kwargs):
    gamma = kwargs["p"]["gamma"]
    c = kwargs["p"]["c"]
    K = np.tanh(gamma * np.dot(x, y) + c)
    return K


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
    b = np.array([0.0])

    # не всегда находит решение из-за ограничения в количестве итераций
    # запускать несколько раз
    ls = cvxopt_solve_qp(P=Pxy, q=q, G=G, h=h, A=A, b=b)
    W, wn = calcW(datasetXY, vector_r, ls)
    supVectorsXY = getSupportVectors(ls, datasetXY)
    t = np.linspace(-5, 5, 100)

    yTrain = np.zeros(2*N)
    yTrain[N:2*N] = np.ones(N)
    xTrain = datasetXY.T
    svc = svm.SVC(kernel='linear', C=20)
    lin_svc = svm.LinearSVC(dual=True, C=20)
    svc.fit(X=xTrain, y=yTrain)  # libsvm
    lin_svc.fit(X=xTrain, y=yTrain)  # liblinear

    fig1 = plt.figure(figsize=(16, 7))
    fig1 = viewFig(fig1, [x, y], 121, "SVC borders",
                   ["SVM quadprog", "SVM qp range"], svc, ["SVC range", "SVC"], supVectorsXY, "lin", ls)
    fig1 = viewFig(fig1, [x, y], 122, "Linear SVC borders",
                   ["SVM quadprog", "SVM qp range"], lin_svc, ["lin SVC range", "lin SVC"], supVectorsXY, "lin", ls)

    # task 3
    my_C = 20
    C = [0.1, 1, 10, my_C]
    Pxz = calculate_P_matrix(datasetXZ, vector_r, kernel_func=np.dot)
    G_withC = np.concatenate((G, np.eye(2 * N)), axis=0)

    D = 1 / (2 * xTrain.var())

    K_limbs = {"lin": [], "poly0": [], "poly1": [], "rad": [], "radGauss": [], "sigmoid": []}
    Pxz_poly0 = calculate_P_matrix(datasetXZ, vector_r, kernel_func=K_poly0, d=3)
    Pxz_poly1 = calculate_P_matrix(datasetXZ, vector_r, kernel_func=K_poly1, d=3)
    Pxz_rad = calculate_P_matrix(datasetXZ, vector_r, kernel_func=K_rad, gamma=1)
    Pxz_radGauss = calculate_P_matrix(datasetXZ, vector_r, kernel_func=K_radGauss, gamma=D)
    Pxz_sigmoid = calculate_P_matrix(datasetXZ, vector_r, kernel_func=K_sigmoid, gamma=0.5, c=-0.01)
    supVectorsXZ = []
    for i in range(0, len(C)):
        h_withC = np.concatenate((h, C[i] * np.ones(2 * N)))
        K_limbs["lin"].append(cvxopt_solve_qp(P=Pxz, q=q, G=G_withC, h=h_withC, A=A, b=b))
        K_limbs["poly0"].append(cvxopt_solve_qp(P=Pxz_poly0, q=q, G=G_withC, h=h_withC, A=A, b=b))
        K_limbs["poly1"].append(cvxopt_solve_qp(P=Pxz_poly1, q=q, G=G_withC, h=h_withC, A=A, b=b))
        K_limbs["rad"].append(cvxopt_solve_qp(P=Pxz_rad, q=q, G=G_withC, h=h_withC, A=A, b=b))
        K_limbs["radGauss"].append(cvxopt_solve_qp(P=Pxz_radGauss, q=q, G=G_withC, h=h_withC, A=A, b=b))
        K_limbs["sigmoid"].append(cvxopt_solve_qp(P=Pxz_poly0, q=q, G=G_withC, h=h_withC, A=A, b=b))  # !!!
    print("Good")

    # расчет опорных векторов для всех случаев
    support_vectors = []
    for value in K_limbs.values():  # for each kernel    K_limbs = dict{ [ [],[],[],[] ] x6}
        K_support = []
        for i in range(0, len(C)):  # for each C
            arr_sv = []
            for j in range(0, len(value[i])):
                if value[i][j] > eps:
                    arr_sv.append(datasetXZ[:, j])
            K_support.append(np.transpose(arr_sv))
        support_vectors.append(K_support)  # support_vectors [     [  [],[],[],[]  ] x6    ]

    # расчет линейных границ и их вывод
    for i in range(0, len(C)):
        W2, wn2 = calcW(datasetXZ, vector_r, K_limbs["lin"][i])
        supVectorsXZ.append(getSupportVectors(K_limbs["lin"][i], datasetXZ))

        xTrain = datasetXZ.T
        svc2 = svm.SVC(kernel='linear', C=C[i])
        svc2.fit(X=xTrain, y=yTrain)  # libsvm

        if i % 2 == 0:
            fig2 = plt.figure(figsize=(16, 7))
        fig2 = viewFig(fig2, [x, z], 121+(i % 2), f"SVC borders with C:{C[i]}",
                       ["SVM quadprog", "SVM qp range"], svc2, ["SVC range", "SVC"], supVectorsXZ[i],
                       "lin", K_limbs["lin"][i])
    print("\n")
    # show()

    # task 4
    # kernel, gamma, coef0, degree
    # dict_params = {"kernel": "poly", "gamma": "scale", "coef0": 0.0, "degree": 3}
    # analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z], "polynomial", support_vectors[1],
    #                   "poly0", K_limbs["poly0"], d=3, kernel=K_poly0)
    #
    # dict_params = {"kernel": "poly", "gamma": "scale", "coef0": 1.0, "degree": 3}
    # analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z], "polynomial not simple", support_vectors[2],
    #                   "poly1", K_limbs["poly1"], d=3, kernel=K_poly1)
    #
    # # "scale" = 1/(n_features * X.var()) , "auto" = 1/n_features
    # dict_params = {"kernel": "rbf", "gamma": 1.0, "coef0": 0.0, "degree": 3}
    # analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z], "radiance func", support_vectors[3],
    #                   "rad", K_limbs["rad"], gamma=1, kernel=K_rad)
    #
    # dict_params = {"kernel": "rbf", "gamma": D, "coef0": 0.0, "degree": 3}
    # analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z], "radiance func Gauss", support_vectors[4],
    #                   "rad Gauss", K_limbs["radGauss"], gamma=D, kernel=K_radGauss)

    dict_params = {"kernel": "sigmoid", "gamma": 0.5, "coef0": -0.01, "degree": 3}
    analiseSVMkernels(C, xTrain, yTrain, dict_params, [x, z], "sigmoid", support_vectors[5],
                      "sigmoid", K_limbs["sigmoid"], gamma=0.5, c=-0.01, kernel=K_sigmoid)

    # параметр C - это компромисс между шириной допустимой полосы и ошибок
    # из рисунков видно, что при увеличении С уменьшается ширина полосы -> в полосе остаются только более значимые
    # опорные вектора -> граница должна строиться лучше -> возможно уменьшение ошибочной классификации

    print("Wow! It is work!")
