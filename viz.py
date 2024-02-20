#%%

import numpy as np
import matplotlib.pyplot as plt

def compute_winding_number(vertices):
    n = len(vertices)
    angles = np.angle(vertices)
    winding_number = (np.sum(np.diff(angles)) + angles[-1] - angles[0]) / (2 * np.pi)
    return winding_number

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

# Define the vertices of the polygon
vertices = np.array([0 + 0j, 0 +1j, 1 +1j, 1 + 0j])
#vertices = np.array([x + np.sin(x) * 1j for x in np.linspace(0, 3 * np.pi, 1000)])
#vertices = np.array([0 + 0j, 1 + 0j, 0.5 + 1j, 0+0j])
#vertices = np.array([1 + 1j, 2 + 2j, 1 + 3j, 0 + 2j, -1 + 3j, -2 + 2j, -1 + 1j, 0 + 0j])


# Apply the DFT to the vertices
vertices_fft = np.fft.ifft(center(vertices), norm = "ortho")
x, y = np.real(center(vertices)), np.imag(center(vertices))

c = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
# Compute the FFT of x and y coordinates separately
fft = np.fft.fft2(c)
vertices_dft = compute_coefficients(center(vertices))


# Compute the winding number of the transformed polygon
#winding_number = compute_winding_number(vertices_fft)

# Plot the transformed polygon
plt.plot(np.real(fft), np.imag(fft), 'bo-')
#plt.plot(np.real(vertices_fft), np.imag(vertices_fft), 'bo-')
#plt.plot(np.real(vertices_dft), np.imag(vertices_dft))
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Transformed Polygon')
plt.grid(True)
plt.show()

print(vertices_fft)

#print(f"Winding Number: {winding_number}")

# %%
#plt.plot(vertices.real, vertices.imag)
magnitudes = np.abs(vertices_fft)
angles = np.angle(vertices_fft)
#plt.plot(vertices_transformed.real, vertices_transformed.imag)
plt.polar(angles, magnitudes)
# %%
c = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

# Compute the FFT using numpy.fft.fft2
fft_result = np.fft.fft(center(vertices))

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(fft_result)

# Plot the magnitude spectrum
plt.imshow(magnitude_spectrum)
plt.colorbar()
plt.title("Magnitude Spectrum")
plt.show()
# %%
