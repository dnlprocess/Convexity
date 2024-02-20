#%%
import numpy as np

def is_convex(vertices):
    # Step 1: Calculate the centroid
    centroid = np.mean(vertices)

    # Step 2: Translate the vertices
    translated_vertices = vertices - centroid

    # Step 3: Sort the translated vertices counterclockwise
    angles = np.angle(translated_vertices)
    sorted_indices = np.argsort(angles)
    sorted_vertices = translated_vertices[sorted_indices]

    # Step 4: Check for counter-clockwise turns
    n = len(sorted_vertices)
    for i in range(n):
        current_vertex = sorted_vertices[i]
        next_vertex = sorted_vertices[(i + 1) % n]
        signed_area = np.imag(np.conj(current_vertex) * next_vertex)
        if signed_area < 0:
            return False

    # Step 5: Polygon is convex if all signed areas are non-negative or zero
    return True

# Example usage
vertices = np.array([1+1j, 2+2j, 3+1j, 2+0.5j])
convex = is_convex(vertices)
print("Convex:", convex)
# %%
