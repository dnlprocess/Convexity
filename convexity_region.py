import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate equilateral triangles in C
x_vals = np.linspace(-10, 10, num=100)
y_vals = np.linspace(-10, 10, num=100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Perform stereographic projection for each triplet of points (a, b, c)
for x in x_vals:
    for y in y_vals:
        a = x + y * 1j
        for i in range(1, 11):
            theta = 2 * np.pi * i / 10
            b = a + np.exp(1j * theta)
            c = a + np.exp(1j * (theta - 2 * np.pi / 3))

            # Perform stereographic projection for a single triplet of points
            def stereographic_projection(z):
                x = (2 * z.real) / (1 + abs(z) ** 2)
                y = (2 * z.imag) / (1 + abs(z) ** 2)
                z = (-1 + abs(z) ** 2) / (1 + abs(z) ** 2)
                return x, y, z
            
            # Perform stereographic projection for each vertex
            ax.plot(*stereographic_projection(a), 'ro')
            ax.plot(*stereographic_projection(b), 'go')
            ax.plot(*stereographic_projection(c), 'bo')

# Set plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Stereographic Projection of Equilateral Triangles')

# Show the plot
plt.show()