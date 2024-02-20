#%%
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fft import fft, fftfreq

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PolygonAnalyzer2d:
    X = None
    centered_X = None
    C_vectors = None
    H_vectors = None
    coefficients = None
    signed_coefficients = None
    eigenvalues = None
    convex = None
    total_angle = None
    simple = None
    mod_vectors = None

    def __init__(self, X, input_type = 'vector'):
        self.C_vectors = self.compute_vectors(X)
        self.H_vectors = self.compute_hermitian_vectors(self.C_vectors)
        if input_type == 'vector':
            self.X = X
            self.centered_X = self.center(self.X)
            self.total_angle, self.convex = self.compute_concavity(self.centered_X)
            self.coefficients = self.compute_coefficients(self.centered_X, self.H_vectors)
        else:
            self.coefficients = X
            X = np.array(np.dot(self.coefficients, self.H_vectors)).ravel()
            self.X = X
            self.centered_X = self.center(self.X)
            self.total_angle, self.convex = self.compute_concavity(self.centered_X)
        self.mod_vectors = self.compute_mod_vectors(self.centered_X)
        self.signed_coefficients = self.coefficients_info(self.coefficients, self.C_vectors)
        self.eigenvalues = self.compute_eigenvalues(self.C_vectors, self.coefficients)
        self.simple = self.is_simple(self.centered_X)

    def compute_theta(self, X1, X2, X3):
        d1 = X2 - X1
        d2 = X3 - X2
        v1 = np.array([np.real(d1), np.imag(d1)])
        v2 = np.array([np.real(d2), np.imag(d2)])

        if np.array_equal(v1, v2):
            return 0,0

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        #if norm_v1 == 0:
        #    print(X1)
        #if norm_v2 == 0:
        #    print(X2)

        #print(X1, X2)
        theta = np.arccos(np.dot(v1,v2)/(norm_v1 * norm_v2))
        cross_product = np.cross(v1, v2)
        if cross_product < 0:
            theta = np.abs(2*np.pi-theta)

        #theta = np.arcsin(cross_product / (norm_v1 * norm_v2))

        return theta, cross_product

    def compute_concavity(self, X):
        vector_angles = np.array([self.compute_theta(X[i], X[(i+1)%len(X)], X[(i+2)%len(X)]) for i in range(len(X))])

        cross_product = np.transpose(vector_angles)[1]
        sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product <= 0)
        
        total_concavity = np.sum(np.transpose(vector_angles)[0])

        return total_concavity, math.isclose(total_concavity, (len(X)-1)*2*np.pi) and sign
    
    def determinant_convexity(self, X):
        cross_product = np.array([self.compute_theta(X[i], X[(i+1)%len(X)], X[(i+2)%len(X)])[1] for i in range(len(X))]).T

        sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product <= 0)
        return cross_product, 1 if sign else -1

        #print(np.array([self.compute_theta(v1, v2)[1] for v1, v2 in zip(X, np.roll(X, -1))]))
        #product_sign = np.prod(np.sign(np.array([self.compute_theta(v1, v2)[1] for v1, v2 in zip(X, np.roll(X, -1))])))
        #print(product_sign)
        #return product_sign*np.abs(product_sign)
    

    def is_simple(self, X):
        n = len(X)

        if n <= 3:
            return True
        
        for i in range(n):
            v1 = np.roll(X, -i)[:2]

            #if np.all(v1 == np.roll(X, -i)):
            #    return False

            for j in range(i + 2, n + i - 1):
                v2 = np.roll(X, -j)[:2]

                #if np.array_equal(v1, v2):
                #    return False

                if self.intersect(v1, v2):
                    return False
        return len(set(X)) == len(X)

    def intersect(self, v1, v2):
        #if np.cross(v1,v2) == 0:
        #    return False
        return self.ccw(v1[0],v2[0],v2[1]) != self.ccw(v1[1],v2[0],v2[1]) and self.ccw(v1[0],v1[1],v2[0]) != self.ccw(v1[0],v1[1],v2[1])

    def ccw(self, A,B,C):
        return (np.imag(C)-np.imag(A)) * (np.real(B)-np.real(A)) > (np.imag(B)-np.imag(A)) * (np.real(C)-np.real(A))

    def compute_coefficients(self, X, hermitian):
        #n = len(X)
        #normalizing_constant = 1/np.sqrt(n)

        #return normalizing_constant *
        #return np.dot(X, hermitian)
        return np.fft.ifft(X, norm='ortho')

    def center(self, X):
        center = np.sum(X, axis=0) / len(X)
        if np.all(np.isclose(center, 0)):
            return X
        return np.array(X - center)

    def compute_vectors(self, X):
        n = len(X)
        w = np.exp(2*np.pi*1j/n)
        j, k = np.meshgrid(range(n), range(n))
        complex_polygon_matrix = (1 / np.sqrt(n))* np.matrix(np.power(w, j*k))

        return complex_polygon_matrix

    def compute_mod_vectors(self, X):
        n = len(X)
        m_values = np.arange(1, n)
        k_values = np.arange(1, n)

        M = np.mod(np.outer(m_values, k_values), n)
        return M.T

    def compute_hermitian_vectors(self, M):
        return np.transpose(M.getH())
    
    def compute_eigenvalues(self, M, X):
        a = np.array([np.dot(X, np.asarray(M[k]).ravel()) / np.dot(np.asarray(M[k]).ravel(), np.asarray(M[k]).ravel()) for k in range(len(X))])
        return np.array([np.sum(np.fromiter((a[j] * (M*np.sqrt(len(a)))[k].T[j] for j in range(len(a))), complex)) for k in range(len(a))])
        #return np.array([np.sum(np.fromiter((coefficients[j] * (M*np.sqrt(len(coefficients)))[k].T[j] for j in range(len(coefficients))), complex)) for k in range(len(coefficients))])

    def move_point(self, delta, index = -1, recenter = False):
        X = self.centered_X

        X[index] = X[index] + delta - delta*1j

        if recenter:
            self.centered_X = self.center(X)
        else:
            self.centered_X = X
        self.total_angle, self.convex = self.compute_concavity(self.centered_X)
        self.C_vectors = self.compute_vectors(self.centered_X)
        self.H_vectors = self.compute_hermitian_vectors(self.C_vectors)
        self.coefficients = self.compute_coefficients(self.centered_X, self.C_vectors)
        self.simple = self.is_simple(self.centered_X)

    def compute_vector_info(self, k):
        v_k = np.asarray(self.C_vectors.T[k]).ravel()

        is_simple = self.is_simple(v_k)
        is_convex = self.compute_concavity(v_k)[1]

        return is_simple, is_convex

    def coefficients_info(self, coefficients, C_vectors):
        sign = lambda k : 1 if self.compute_concavity(np.asarray(C_vectors.T[k]).ravel())[1] else -1
        signs = np.array([sign(k) for k in range(len(np.asarray(coefficients).ravel()))])
        #return signs, np.asarray(coefficients).ravel().T
        return np.matrix([signs[k] * np.asarray(coefficients).ravel()[k] for k in range(len(np.asarray(coefficients).ravel()))])
        
        
        #np.matrix(np.array([sign(k) * np.asarray(coefficients).ravel()[k] for k in range(len(np.asarray(coefficients).ravel()))]))
        
    def reset_coefficients(self, coefficients):
        self.coefficients = coefficients
        self.recompute_X(self.coefficients)

    def recompute_X(self, coefficients):
        X = np.dot(coefficients, self.H_vectors)
        self.X = X
        self.centered_X = self.center(self.X)
        return self.centered_X

    def visualize(self, name = ""):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        cmap = cm.get_cmap('tab10')
        color = cmap(0)

        handles = []
        colors = []
        labels = []


        n = len(self.centered_X)

        # Plotting the arrows between each point
        for i in range(n):
            v1 = self.centered_X[i]
            v2 = self.centered_X[(i + 1) % n]
            ax.arrow(np.real(v1), np.imag(v1), np.real(v2) - np.real(v1), np.imag(v2) - np.imag(v1),
                    head_width=0.01, head_length=0.01, fc=color, ec=color)

        # Compute and display concavity information
        total_concavity, is_convex = self.compute_concavity(self.centered_X)
        concavity_info = f"Shape {i + 1}, {name} - Concavity: {total_concavity/np.pi:.4f} * π | Convex: {is_convex}"
        line, = ax.plot([], [], color='blue')
        handles.append(line)
        labels.append(concavity_info)
        ax.legend(handles, labels, loc='upper left')

    def visualize_coefficients(self, coefficients):
        fig, ax = plt.subplots()

        # Color code for coefficients based on properties
        color_simple = 'blue'
        color_convex = 'red'
        color_simple_convex = 'purple'
        color_default = 'green'

        handles = []
        labels = []
        colors = []

        coefficients = np.asarray(coefficients).ravel()

        real_parts = np.real(coefficients)
        imag_parts = np.imag(coefficients)

        for k, v_k in enumerate(self.C_vectors.T):
            v_k = np.asarray(v_k).ravel()
            is_simple = self.is_simple(v_k)
            total_concavity, is_convex = self.compute_concavity(v_k)

            if is_simple and is_convex:
                color = color_simple_convex
            elif is_simple:
                color = color_simple
            elif is_convex:
                color = color_convex
            else:
                color = color_default

            #print("real")
            #print(real_parts[k])
            ax.scatter(real_parts[k], imag_parts[k], c=color)
            ax.annotate(str(k), (real_parts[k], imag_parts[k]))
            handles.append(ax.plot([], [], color=color)[0])
            colors.append(color)
            labels.append(f"Coefficient {k} - Convex: {is_convex}, {total_concavity/np.pi} | Simple: {is_simple} | Location: ({round(real_parts[k], 2)}, {round(imag_parts[k], 2)})")

        colors.append('black')
        handles.append(ax.plot([], [], color='black')[0])
        labels.append(f"Shape - Convex {self.convex}")
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.legend(handles, labels, labelcolor = colors, loc='upper left')
    
    def visualize_C_vectors(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        cmap = cm.get_cmap('tab10')

        handles = []
        colors = []
        labels = []

        n = len(self.centered_X)

        for k, v_k in enumerate(self.C_vectors.T):
            #color = cmap(k % 10)
            v_k = np.asarray(v_k).ravel()

            is_simple = self.is_simple(v_k)
            total_angle, is_convex = self.compute_concavity(v_k)

            if is_simple and is_convex:
                color = 'purple'
            elif is_simple:
                color = 'blue'
            elif is_convex:
                color = 'red'
            else:
                color = 'green'

            # Plotting the arrows between each point
            for j in range(n):
                v1 = v_k[j]
                v2 = v_k[(j + 1) % n]  # Wrap around to the first point for the last arrow
                ax.arrow(np.real(v1), np.imag(v1), np.real(v2) - np.real(v1), np.imag(v2) - np.imag(v1),
                        head_width=0.1, head_length=0.1, fc=color, ec=color)

            # Compute and display concavity information
            total_concavity, is_convex = self.compute_concavity(v_k)
            is_simple = self.is_simple(v_k)
            #concavity_info = f"Concavity: {total_concavity:.2f} (<= 2π: {is_convex})"            #handles.append(concavity_info)
            concavity_info = f"Concavity: {total_concavity:.2f} (<= 2π: {is_convex})"
            handles.append(ax.plot([], [], color=color)[0])
            colors.append(color)
            labels.append(f"Vector {k + 1} - Concavity: {total_concavity/np.pi:.4f} * π | Convex: {is_convex} | Simple: {is_simple}")

        ax.legend(handles, labels, labelcolor = colors, loc='upper right')
# %%
alpha = np.array([0,1+0j,0,0,0])
n = len(alpha)
omega = lambda n: np.exp(1j*2*np.pi/n)
pentagon = PolygonAnalyzer2d(np.array([0,1,0,0,0]), input_type='coeff')
#pentagon = PolygonAnalyzer2d(np.array([3+3j,2 + 1j, 3 + 1.2j, 2.2 + 4j, 1+1.2j]))
#alpha = pentagon.coefficients
v = pentagon.C_vectors
v_inverse = pentagon.H_vectors
identity = np.identity(len(alpha))
permutation = np.flipud(identity)
v_squared = np.linalg.matrix_power(v, 2)
diff_1 = (permutation @ v_squared - identity)
diff_2 = (permutation @ v_squared - identity) @ permutation
difference_1 = diff_1 @ v
difference_2 = diff_2 @ v
diag_1 = np.diag(np.array([omega(n)**k -1 for k in range(n)]))
diag_2 = np.diag(np.array([omega(n)**((n-2)*k) - omega(n)**((n-1)*k) for k in range(n)]))

def calculate_norm_result(alpha):
    #f1 = np.dot(difference_1, alpha)
    #f2 = np.dot(difference_2, alpha)
    simple = np.array([omega(n)**k for k in range(n)])
    return np.sum(simple@alpha)

    #f1 = np.dot(difference_1, np.conjugate(alpha))
    #f2 = np.dot(difference_2, np.conjugate(alpha))

    #f1 = np.dot(v@diag_1, alpha)
    f1 = v@diag_1@alpha
    #f2 = np.dot(v_inverse@diag_2, np.conjugate(alpha))
    f2 = v_inverse@diag_2@np.conjugate(alpha)

    result = np.multiply(f1, f2)
    #return np.linalg.norm(v_inverse@np.multiply(diag_1@alpha, diag_2@np.conjugate(alpha)))
    #result = np.multiply(np.dot(diag_1, alpha), np.dot(diag_2, np.conjugate(alpha)))
    #result = np.dot(v_squared@diag_1, np.conjugate(alpha))/np.dot(diag_2, alpha)
    #result = np.multiply(np.dot(v@diag_1, alpha), np.dot(v@diag_1, alpha))
    #result = np.dot(f1,f2.T)
    #result = np.multiply(np.linalg.matrix_power(permutation @ v_squared - identity,2), np.multiply(v@alpha, permutation@v.getH()@alpha.T))
    result = np.asarray(result).ravel()
    result = result/(np.linalg.norm(f1)*np.linalg.norm(f2))
    return result
    #result = np.arcsin(result)
    norm = result/np.linalg.norm(result)
    #return np.imag(result[4]) >= 0
    #return np.angle(np.min(norm))
    """if np.any(np.imag(result)<):
        return 1
    return 0"""
    """final = np.min(np.real(result))
    if -0.001 <= final <= 0.001:
        return final
    return 0
    
    z_min = 0.5
    z_max = 1.5
    indices = np.where((z >= z_min) & (z <= z_max))

    filtered_z = z[indices]
    
    """
    return np.angle(result[np.argmin(np.abs(result))])<0 #and np.abs(np.sum(np.angle(result))/(2*np.pi)) <0.1
    
    return np.imag(result[np.argmin(np.abs(result))])
    return np.min(np.abs(result))
    return np.abs(np.angle(result[np.argmin(np.abs(result))]))
    return np.min(np.angle(result))
    #return np.all(np.imag(result)0) 
    #and 
    return np.abs(np.sum(np.angle(result))/(2*np.pi))
    #return np.sum(np.angle(result))
    return np.min(np.abs(np.angle(result)))
    return np.min(np.abs(np.angle(result)))<= 0.2
    return np.min(result[np.abs(np.angle(result)) <=0.1])
    return np.imag(result[np.argmin(np.abs(result))])
    return np.min(np.abs(result))
    #return np.sum(np.imag(result))
    return np.min(np.abs(result))
    #return np.abs(np.angle(np.min(np.imag(result))))<= 0.2
    #return np.min(np.abs(np.angle(result)))<= 0.2
    #return np.any(np.abs(np.angle(result))<=0.2)
    #return result[np.abs(np.min(np.imag(result)))<1]
    """

    threshold = 0.1+0.1j
    
    within_range = np.any((result.real <= threshold.real) & (result.imag <= threshold.imag))
    return within_range.astype(int)"""

# Define the range of values for the coefficient (alpha)
limit = 2
real_values = np.linspace(-limit, limit, 100)
imag_values = np.linspace(-limit, limit, 100)
alpha_meshgrid = np.array(np.meshgrid(real_values, imag_values)).T.reshape(-1, 2)

norm_results = []
for alpha_value in alpha_meshgrid:
    x = alpha_value[0]
    y = alpha_value[1]
    alpha[3] = x + y * 1j
    #alpha[3] = (1/2)*(np.sqrt(x + y*1j)) - 0.2 + 0.08*1j
    #alpha[4] = x**2 + (x - 0.3 * y)**2 * 1j - 2*x + y**(9/4)*1j
    norm_result = calculate_norm_result(alpha)
    norm_results.append(norm_result)

norm_results_grid = np.array(norm_results).reshape(len(real_values), len(imag_values))
real_grid, imag_grid = np.meshgrid(real_values, imag_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(real_grid, imag_grid, norm_results_grid, cmap=plt.colormaps["twilight_shifted"])

ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_zlabel('Norm Result')

plt.title('Norm Result vs. Alpha Coefficients')
plt.show()
# %%
