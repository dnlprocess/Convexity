#%%

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from vectorConvexity import PolygonAnalyzer2d


triangle = np.array([1 + 1j, 2 + 3j, 3 + 1j])
square = np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
double_square = 1.2 * np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j, 1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
oblong_pentagon = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j])
concave_shape = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j, 1 + 2j])
pentagon = 2 * np.exp(1j * np.linspace(0, 2 * np.pi, 6))[:-1]
hexagon = 2 * np.exp(1j * np.linspace(0, 2 * np.pi, 7))[:-1]



#shapes = [triangle, square, double_square, pentagon, oblong_pentagon, concave_shape, hexagon]
#shapes = [PolygonAnalyzer2d(shape) for shape in shapes]
oblong_square = np.array([1 + 1j, 1 + 2j, 2 + 2j, 2.5 + 0.5j])
oblong_square = PolygonAnalyzer2d(oblong_square)


def determine_color(polygon, k=0, t='self'):
    if t == 'vector': simple, convex = polygon.compute_vector_info(k+1)
    else: simple, convex = polygon.simple, polygon.convex
    color = None
    if simple and convex:
        color = 'purple'
    elif simple:
        color = 'blue'
    elif convex:
        color = 'red'
    else:
        color = 'green'
    return color


def visualize_polygon_coefficient_change(polygon, delta, delta_t = 1000):
    n = len(polygon.X)
    t = np.arange(0, np.abs(delta), np.abs(delta)/delta_t)
    c = polygon.convex
    s = polygon.simple

    norm_coefficients = np.array([])
    coefficients_x = np.array([])
    coefficients_y = np.array([])
    info = [(0, determine_color(polygon))]
    for X in t:
        polygon.move_point(delta*(1/delta_t), recenter = True)
        current_coefficients = np.asarray(polygon.coefficients).ravel()[1:]
        norms = np.array([np.linalg.norm(np.array(np.real(c), np.imag(c))) for c in current_coefficients])
        norm_coefficients = np.append(norm_coefficients, norms.T)
        coefficients_x = np.append(coefficients_x, np.real(current_coefficients).T)
        coefficients_y = np.append(coefficients_y, np.imag(current_coefficients).T)
        if polygon.convex != c:
            info.append((X, determine_color(polygon)))
            print((X, determine_color(polygon)))
            c = polygon.convex
        if polygon.simple != s:
            print((X, determine_color(polygon)))
            info.append((X, determine_color(polygon)))
            s = polygon.convex
    norm_coefficients = norm_coefficients.reshape((n-1, delta_t), order = 'F')
    coefficients_x = coefficients_x.reshape((n-1, delta_t), order = 'F')
    coefficients_y = coefficients_y.reshape((n-1, delta_t), order = 'F')


    fig1, ax = plt.subplots()
    for k, c_k in enumerate(coefficients_x):
        color = determine_color(polygon, k, 'vector')
        b_color = determine_color(polygon, k)
        ax.plot(t, c_k, label=f'c_{k+1}x', color = color)
        ax.annotate(str(k+1), (t[-1], c_k[-1]))

    """fig2, y = plt.subplots()
    for k, c_k in enumerate(coefficients_y):
        color = determine_color(polygon, k)
        y.plot(t, c_k, label=f'c_{k+1}y')

    fig3, n = plt.subplots()
    for k, c_k in enumerate(norm_coefficients):
        color = determine_color(polygon, k)
        n.plot(t, c_k, label=f'c_{k+1}_n')"""

    plt.xlabel("time")
    plt.ylabel("norm of c_k")
    
    plt.title(f"Moving polygon vertex by {delta}")
    plt.legend()

    polygon.visualize()
    plt.show()

oblong_square.visualize()
oblong_square.visualize_coefficients(oblong_square.coefficients)
visualize_polygon_coefficient_change(oblong_square, -2)
oblong_square.visualize_coefficients(oblong_square.coefficients)
# %%
