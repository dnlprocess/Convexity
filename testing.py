#%%
import numpy as np
import matplotlib.pyplot as plt

def total_angle(v):
    n = len(v)
    a = np.fft.ifft(v, norm='ortho')
    
    complex_sum = np.sum(np.abs(a * np.exp(2j * np.pi * np.arange(1, len(a) + 1) / len(a))))

    # Calculate the total angle as the argument of the complex sum
    total_angle = np.angle(complex_sum)

    # Calculate the total angle the usual way
    total_angle_usual = 0.0
    for i in range(n):
        v1 = v[i]
        v2 = v[(i + 1) % n]
        total_angle_usual += np.abs(np.angle(v2 - v1))

    return total_angle, total_angle_usual, a

def plot_polygon(v):
    n = len(v)
    x = np.real(v)
    y = np.imag(v)

    # Plot the polygon
    plt.figure(figsize=(8, 8))
    plt.plot(np.append(x, x[0]), np.append(y, y[0]), marker='o', linestyle='-', color='blue', label='Polygon')
    
    # Plot the Fourier coefficients
    plt.scatter(range(n), np.abs(eigenvalues), color='red', label='Fourier Coefficients')
    
    plt.title('Polygon and Fourier Coefficients')
    plt.xlabel('Vertex Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
v = np.array([1, 2, 3 + 2j, 1 + 2j, 0.5 +0.6j])
result_eigenvalues, result_usual, eigenvalues = total_angle(v)

print("Total Angle of Polygon (Eigenvalues):", result_eigenvalues)
print("Total Angle of Polygon (Usual):", result_usual)

# Plot the polygon and Fourier coefficients
plot_polygon(v)
# %%
