#%%

import numpy as np
import matplotlib.pyplot as plt

def compute_coefficients(X):
    n = len(X)
    w = np.e**(2*np.pi*1j*(1/n))
    j, k = np.meshgrid(range(n), range(n))
    complex_polygon_matrix = np.power(w, j*k)
    normalizing_constant = 1/np.sqrt(n)

    return normalizing_constant * np.dot(complex_polygon_matrix, np.transpose(X))

def center(X):
    center = np.sum(X, axis=0) / len(X)
    #if np.real(X - center) == np.zeros(len(X)):
    #    return
    #else:
    return X - center

#x = np.array([0.+0.j, 2.+1.j, 3.+5.j, 0.+6.j, -3.+5.j, -2.+3.j, 0.+0.j], dtype=complex)
#n = len(x)-1

#F = np.array([np.sum(x[k] * np.exp(-2j * np.pi * np.arange(n) * k / n)) / np.sqrt(n) for k in range(n)], dtype=complex)

triangle = np.array([1 + 1j, 2 + 3j, 3 + 1j])
square = np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
double_square = 1.2 * np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j, 1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
oblong_pentagon = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j])
convex_shape = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j, 1 + 2j])

shapes = [triangle, square, oblong_pentagon, double_square, convex_shape]
centered_shapes = [center(shape) for shape in shapes]
coefficients = [compute_coefficients(shape) for shape in centered_shapes]

#%%
#plt.plot(x.real, x.imag)
plt.plot(F.real, F.imag)
# %%
