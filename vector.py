#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_theta(v1, v2):
    v1 = np.array([np.real(v1), np.imag(v1)])
    v2 = np.array([np.real(v2), np.imag(v2)])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cross_product = np.cross(v1, v2)

    theta = np.arcsin(cross_product / (norm_v1 * norm_v2))

    #theta = np.arccos(np.dot(v1,v2)/(norm_v1 * norm_v2))
    return theta, cross_product

def compute_concavity(X):
    vector_angles = np.array([compute_theta(v1, v2) for v1, v2 in zip(X, np.roll(X, -1))])

    cross_product = np.transpose(vector_angles)[1]
    sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product < 0)
    
    total_concavity = np.sum(np.transpose(vector_angles)[0])

    return total_concavity, (total_concavity <= 2*np.pi or math.isclose(total_concavity, 2*np.pi)) and sign
#%%


def is_simple(X):
    n = len(X)

    if n <= 3:
        return True
    
    for i in range(n):
        v1 = np.roll(X, -i)[:2]

        for j in range(i + 2, n + i - 1):
            v2 = np.roll(X, -j)[:2]

            print("i: " + str(j) + ", j: " + str(j))
            print(v1, v2)

            if intersect(v1, v2):
                return False
    return True

def intersect(v1, v2):
    #if np.cross(v1,v2) == 0:
    #    return False
    return ccw(v1[0],v2[0],v2[1]) != ccw(v1[1],v2[0],v2[1]) and ccw(v1[0],v1[1],v2[0]) != ccw(v1[0],v1[1],v2[1])

def ccw(A,B,C):
    return (np.imag(C)-np.imag(A)) * (np.real(B)-np.real(A)) > (np.imag(B)-np.imag(A)) * (np.real(C)-np.real(A))

#%%







def compute_coefficients(X):
    n = len(X)
    w = np.e**(2*np.pi*1j*(1/n))
    j, k = np.meshgrid(range(n), range(n))
    complex_polygon_matrix = np.power(w, j*k)
    normalizing_constant = 1/np.sqrt(n)

    return normalizing_constant * np.dot(complex_polygon_matrix, np.transpose(X))

def center(X):
    center = np.sum(X, axis=0) / len(X)
    if np.all(np.isclose(center, 0)):
        return X
    return X - center

def compute_vectors(X):
    n = len(np.array(X))
    w = np.e**(2*np.pi*1j*(1/n))
    j, k = np.meshgrid(range(n), range(n))
    complex_polygon_matrix = np.power(w, j*k)

    c_vectors = []
    for col in range(complex_polygon_matrix.shape[1]):
        c_vectors.append(np.array(complex_polygon_matrix[:, col]))
    
    return complex_polygon_matrix, c_vectors


def compute_hermitian_vectors(M):
    hermitian = np.array(np.transpose(M.getH()))

    hermitian_vectors = []
    for col in range(hermitian.shape[1]):
        shapes.append(np.array(hermitian[:, col]))

    return hermitian_vectors

#%%
def visualize(shapes):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cmap = cm.get_cmap('tab10')

    handles = []
    colors = []
    labels = []

    for i, shape in enumerate(shapes):
        X = shape
        n = len(X)

        # Plotting the arrows between each point
        for j in range(n):
            v1 = X[j]
            v2 = X[(j + 1) % n]  # Wrap around to the first point for the last arrow
            color = cmap(i)  # Get a unique color for each shape
            ax.arrow(np.real(v1), np.imag(v1), np.real(v2) - np.real(v1), np.imag(v2) - np.imag(v1),
                    head_width=0.1, head_length=0.1, fc=color, ec=color)

        # Compute and display concavity information
        total_concavity, is_convex = compute_concavity(X)
        #concavity_info = f"Concavity: {total_concavity:.2f} (<= 2π: {is_convex})"            #handles.append(concavity_info)
        concavity_info = f"Concavity: {total_concavity:.2f} (<= 2π: {is_convex})"
        handles.append(ax.plot([], [], color=cmap(i))[0])
        colors.append(cmap(i))
        labels.append(f"Shape {i + 1} - Concavity: {total_concavity/np.pi:.4f} * π | Convex: {is_convex}")

    ax.legend(handles, labels, labelcolor = colors, loc='upper left')
    plt.show()

pentagon = pentagon = 2 * np.exp(1j * np.linspace(0, 2 * np.pi, 6))
oblong_pentagon = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j])
shapes = [pentagon, oblong_pentagon, np.array([ 1. +0.00000000e+00j, -0.5+8.66025404e-01j, -0.5-8.66025404e-01j,
   1. -8.32667268e-16j, -0.5+8.66025404e-01j, -0.5-8.66025404e-01j])]
visualize(shapes)
#%%

triangle = np.array([1 + 1j, 2 + 3j, 3 + 1j])
square = np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
double_square = 1.2 * np.array([1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j, 1 + 1j, 1 + 2j, 2 + 2j, 2 + 1j])
oblong_pentagon = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j])
concave_shape = np.array([1 + 1j, 2 + 4j, 4 + 3j, 3 + 1j, 2 + 2j, 1 + 2j])
pentagon = np.array([1 + v + v * 1j for v in np.linspace(0, 2 * np.pi, 5)])
hexagon = np.array([1 + v + v * 1j for v in np.linspace(0, 2 * np.pi, 6)])

shapes = [triangle, square, oblong_pentagon, double_square, convex_shape]
shapes = [pentagon, oblong_pentagon]
shapes = [center(shape) for shape in shapes]
coefficients = [compute_coefficients(shape) for shape in shapes]
fft_coefficients = [np.fft.ifft(shape, norm = "ortho") for shape in shapes]
concavities = [compute_concavity(shape) for shape in shapes]

visualize(centered_shapes)
visualize(coefficients)
#analyzer.visualize(fft_coefficients)
#vectors = [analyzer.visualize_vectors(shape) for shape in shapes]

for i, (shape, (total_concavity, is_convex)) in enumerate(zip(shapes, concavities)):
    print(f"Shape {i + 1} - Concavity: {total_concavity:.4f} | Convex: {is_convex}")


# %%
theta = np.linspace(0, 2 * np.pi, 4)
pentagon = 2 * np.exp(1j * np.linspace(0, 2 * np.pi, 4))