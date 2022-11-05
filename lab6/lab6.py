import lab4.lab5 as labs
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')
    u = np.array([0, 0])
    v = np.array([1, 1])
    res = labs.d(u, v)
    print(res)
