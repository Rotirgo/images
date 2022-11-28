import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show
import lab1_2_4.lab124 as lab1


def Parzen_classificator(x, train_classes, train_B):
    n_classes = len(train_classes)

    f = np.zeros((n_classes, ))
    P = np.zeros((n_classes, ))

    cnt = 0
    k = 0.25
    for i in range(0, n_classes):
        n_vectors = len(np.transpose(train_classes[i]))
        h = n_vectors**(-k / n)
        sum = 0
        const = pow((2.0 * np.pi), (-n / 2.0)) * pow(h, -n) * pow(np.linalg.det(train_B[i]), -0.5)
        for x_i in np.transpose(train_classes[i]):
            dist = np.matmul(np.matmul((x-x_i), np.linalg.inv(train_B[i])), (x-x_i))
            power = -0.5 * pow(h, -2) * dist
            sum += const * np.exp(power)
        sum /= n_vectors
        cnt += n_vectors

        f[i] = sum
        P[i] = n_vectors
    P /= cnt

    table = np.eye(n_classes).astype(bool)
    for i in range(0, n_classes - 1):
        for j in range(i + 1, n_classes):
            if P[i] * f[i] >= P[j] * f[j]:
                table[i][j] = True
                table[j][i] = False
            else:
                table[i][j] = False
                table[j][i] = True

    for i in range(0, len(table)):
        if (table[i] == np.ones(len(table)).astype(bool)).all():
            return i


def d(x, z):
    dist = np.sum(np.square(x-z))
    return np.sqrt(dist)


Distance = np.vectorize(d, signature='(n),(m)->()')  # берет одноморный массив из превого аргумента, одномерный массив
# из второго аргумента, на выходе число. и так по всем одномерным массивам из первого аргумента.


def K_neighbors_classificator(x, train_classes, K):
    all_vectors = np.transpose(np.concatenate(train_classes, axis=1))
    r = np.zeros((len(np.transpose(train_classes[0])), ))

    for i in range(1, len(train_classes)):
        size = np.shape(train_classes[i])
        r = np.concatenate([r, np.ones((size[1], ))*i])

    distances = Distance(all_vectors, x)
    neighbors = []
    for i in range(0, K):
        idx_min = np.argmin(distances)
        neighbors.append(idx_min)
        distances[idx_min] = distances.max()
    neighbors_classes = list(r[neighbors])
    num_class = max(set(neighbors_classes), key=neighbors_classes.count)
    return int(num_class)


def get_classes(test_X, train_X, func, **kwargs):
    classes = []
    for i in range(0, len(train_X)):
        classes.append(np.empty((n, 1)))

    for el in np.transpose(np.concatenate(test_X, axis=1)):
        num_class = -1
        if func == Parzen_classificator:
            arr_B = kwargs["B"]
            num_class = func(el, train_X, arr_B)  # номер классов идет с 0
        elif func == K_neighbors_classificator:
            K = kwargs["K"]
            num_class = func(el, train_X, K)  # номер классов идет с 0
        else:
            arr_M = kwargs["M"]
            arr_B = kwargs["B"]
            num_class = lab1.BayeslassificatorB(el, arr_M, arr_B, 1)  # номер классов идет с 0
        classes[num_class] = np.concatenate([classes[num_class], np.reshape(el, (n, 1))], axis=1)

    for i in range(0, len(train_X)):
        classes[i] = classes[i][:, 1:]
    return classes


def view_classes(true_classes, title, *args):
    if len(args) > 0:
        fig = plt.figure(figsize=(16, 7))
        plt.title(title)
        fig.add_subplot(121)
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.title(title)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-1.5, 3.5)
    c = ['r.', 'gx', 'b+']
    for i in range(0, len(true_classes)):
        plt.plot(true_classes[i][0], true_classes[i][1], c[i], label=f"class {i}")
    plt.legend()

    if len(args) > 0:
        classes = args[0]

        fig.add_subplot(122)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-1.5, 3.5)
        c = ['r.', 'gx', 'b+']
        for i in range(0, len(classes)):
            plt.plot(classes[i][0], classes[i][1], c[i], label=f"class {i}")

        errors = calc_errors(classes, true_classes)
        errors = np.transpose(errors)
        plt.scatter(errors[0], errors[1], s=90, linewidth=1, facecolors='none',
                    edgecolors='orange', label="Ошибочная классификация", alpha=0.7)
    plt.legend()


def calc_errors(classes, true_classes):
    errors = []
    for i in range(0, len(classes)):  # в каждом классе
        for el in np.transpose(classes[i]):  # идем по векторам
            if not (el == np.transpose(true_classes[i])).all(axis=1).any():  # сверяем наличие в истинных классах
                errors.append(el)
    return errors


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


M1 = [0, 0]
M2 = [1, 1]
M3 = [-1, 1]
B1 = [[0.12, 0.0],
      [0.0, 0.12]]
B2 = [[0.25, -0.1],
      [-0.1, 0.17]]
B3 = [[0.25, 0.1],
      [0.1, 0.22]]

n = 2
N = 200


if __name__ == '__main__':
    print_hi('PyCharm')

    A1 = lab1.calcMatrixA(B1)
    A2 = lab1.calcMatrixA(B2)
    A3 = lab1.calcMatrixA(B3)

    train_N = 100
    X_train = lab1.generate_vectors(A1, M1, n, train_N)
    Y_train = lab1.generate_vectors(A2, M2, n, train_N)
    Z_train = lab1.generate_vectors(A3, M3, n, train_N)

    X_test = lab1.generate_vectors(A1, M1, n, N)
    Y_test = lab1.generate_vectors(A2, M2, n, N)
    Z_test = lab1.generate_vectors(A3, M3, n, N)

    train = [X_train, Y_train, Z_train]
    test = [X_test, Y_test, Z_test]
    M = [M1, M2, M3]
    B = [B1, B2, B3]

    Parzen_classes = get_classes(test, train, Parzen_classificator, B=B)
    Parzen_errors = calc_errors(Parzen_classes, test)
    print(f"Суммарная вероятность ошибочной классификации Парзена: {100 * len(Parzen_errors) / (len(test) * N):.2f}%")

    view_classes(train, "Тренировочные классы")
    view_classes(test, "Классификация Парзена", Parzen_classes)

    K = [1, 3, 5]
    for k in K:
        K_nighbors_classes = get_classes(test, train, K_neighbors_classificator, K=k)
        K_nighbors_errors = calc_errors(K_nighbors_classes, test)
        print(f"Суммарная вероятность ошибочной классификации {k} ближайших "
              f"соседей: {100 * len(K_nighbors_errors) / (len(test) * N):.2f}%")
        view_classes(test, f"Классификация {k} ближайших соседей", K_nighbors_classes)

    Bayess_classes = get_classes(test, train, func="Bayess", M=M, B=B)
    Bayess_errors = calc_errors(Bayess_classes, test)

    print(f"Суммарная вероятность ошибочной классификации Парзена: {100 * len(Bayess_errors) / (len(test) * N):.2f}%")
    view_classes(test, f"Байессовский классификатор", Bayess_classes)

    show()
