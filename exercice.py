#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import integrate


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([math.sqrt(cartesian_coordinates[0] ** 2 + cartesian_coordinates[1] ** 2),
                     np.arctan(cartesian_coordinates[1] / cartesian_coordinates[0])])


def find_closest_index(values: np.ndarray, number: float) -> int:
    if number < values[0]:
        return 0
    elif number > values[-1]:
        return len(values) - 1

    for index, elem in enumerate(values):
        if elem == number:
            return np.where(values == values[index - 1])[0][0]
        else:
            return np.where(values == values[index - 1])[0][0]


def plot():
    x = np.linspace(-1, 1, 250)
    y = x ** 2 * np.sin(1/(x ** 2)) + x

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot2():
    # x = np.linspace(-4, 4, 250)
    # y = x ** 2 * np.sin(1/(x ** 2)) + x
    #
    def f(x):
        return np.exp(-x ** 2)
    integrale = integrate.quad(f, -np.inf, np.inf)
    print(integrale)

    ix = np.linspace(-4, 4, 250)
    iy = f(ix)

    fig, ax = plt.subplots()
    ax.plot(ix, iy, 'r', linewidth=2)
    ax.set_ylim(bottom=0)
    ax.fill_between(ix, 0, iy, facecolor='red')
    plt.show()

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    #print(linear_values())
    #print(coordinate_conversion(np.array([1, 5])))
    #print(find_closest_index(np.array([1, 10, 100, 1000, 10000, 100000]), -1))
    #plot()
    plot2()