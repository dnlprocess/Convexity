from vectorConvexity import PolygonAnalyzer2d
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.widgets import Slider

convexity_counter = False
edge_shapes = []

def on_click(event, coefficients, c, coeff_sliders):
    if event.inaxes is not None:
        coefficients[c] = event.xdata + 1j * event.ydata
        # Update the corresponding slider values
        coeff_sliders[c][0].set_val(np.real(coefficients[c]))
        coeff_sliders[c][1].set_val(np.imag(coefficients[c]))
        shape = PolygonAnalyzer2d(coefficients, input_type='coefficients')
        shape.visualize()
        plt.show()

def update_plot(fig, x, y, n, o, c, contours_convex, contours_simple):
    convex_part, simple_part, regular_part = complexity(x, y, n, o, c)
    contours_convex.set_array(convex_part.flatten())
    contours_simple.set_array(simple_part.flatten())
    fig.canvas.draw_idle()
 

def plot(n, o, c):
    coeff_range1 = np.linspace(-1.8,1.8,200)
    coeff_range2 = np.linspace(0, 2*np.pi,200)
    x, y = np.meshgrid(coeff_range1, coeff_range2)

    convex_part, simple_part, regular_part = complexity(x, y, n, o, c)

    fig, ax = plt.subplots()
    contours_convex = ax.contourf(x, y, convex_part, alpha=0.5, cmap='Reds')
    contours_simple = ax.contourf(x, y, simple_part, alpha=0.5, cmap='Blues')
    #contours_regular = ax.contourf(x, y, regular_part, alpha=0.5, cmap='Oranges')

    # Add colorbar
    #cbar = fig.colorbar(contours_convex, ax=ax)
    #cbar = fig.colorbar(contours_simple, ax=ax)
    #cbar = fig.colorbar(contours_regular, ax=ax)
    #cbar.set_label('Colorbar')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')

    plt.axis('scaled')

    coefficients = np.zeros(n, dtype='complex')
    for pair in o:
        coefficients[pair[0]] = pair[1]
    X = PolygonAnalyzer2d(coefficients, input_type='coeff')

    coeff_sliders = []  # List to hold slider objects
    for i in range(len(coefficients)):
        real_slider = plt.axes([0.25, 0.01 + i*0.03, 0.65, 0.02])
        imag_slider = plt.axes([0.25, 0.02 + i*0.03, 0.65, 0.02])
        real_val = plt.Slider(real_slider, 'Real', -10, 10, valinit=np.real(coefficients[i]))
        imag_val = plt.Slider(imag_slider, 'Imaginary', -10, 10, valinit=np.imag(coefficients[i]))
        coeff_sliders.append((real_val, imag_val))
    
    def update_sliders(coeff_sliders, coefficients, o, c, x, y, n, contours_convex, contours_simple):
        for i, (real_val, imag_val) in enumerate(coeff_sliders):
            if i != c:  # Only update non-changing coefficient sliders
                coefficients[i] = real_val.val + 1j * imag_val.val
        o[c][1] = coeff_sliders[c].val  # Update the changing coefficient in the o array
        update_plot(fig, x, y, n, o, c, contours_convex, contours_simple)  # Call update_plot to update the plot based on the new coefficients


    for i, (real_val, imag_val) in enumerate(coeff_sliders):
        if i != c:  # Only connect non-changing coefficient sliders
            real_val.on_changed(lambda val: update_sliders(coeff_sliders, coefficients, o, c, x, y, n, contours_convex, contours_simple))
            imag_val.on_changed(lambda val: update_sliders(coeff_sliders, coefficients, o, c, x, y, n, contours_convex, contours_simple))


    on_click_partial = partial(on_click, coefficients=coefficients, c=c, coeff_sliders=coeff_sliders)
    fig.canvas.mpl_connect('button_press_event', on_click_partial)
    fig.savefig('filename.eps', format='eps')
    plt.show()

def complexity(x, y, n, o, c):
    if np.isscalar(x) and np.isscalar(y):
        return compute_complexity(x, y, n, o, c)
    
    result_convex = np.zeros_like(x)
    result_simple = np.zeros_like(x)
    result_regular = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result_convex[i,j], result_simple[i,j], result_regular[i,j] = compute_complexity(x[i, j], y[i, j], n, o, c)
    
    return result_convex, result_simple, result_regular

def compute_complexity(x, y, n, o, c):
    coefficients = np.zeros(n, dtype='complex')
    for pair in o:
        coefficients[pair[0]] = pair[1]
    coefficients[c] = x * np.exp(1j* y)
    #try changing polar of coefficeints[c]
    #coefficients[c+1] = 2*x + 2*y*1j
    #coefficients[int(np.ceil(1.6*c))] = 0+y * 1j
    #coefficients[int(np.ceil(2.1*c))] = x + 0j
    X = np.fft.fft(coefficients, norm='ortho')

    convex, regular = is_convex(X)

    #evolved_polygon = PolygonAnalyzer2d(flow(X, 100), input_type='coefficients')
    #regular = 1 if evolved_polygon.total_angle == 0 else -1

    global convexity_counter
    if convex != convexity_counter:
        edge_shapes.append((X, convex))
        convexity_counter = convex

    convex = 1 if convex else -1
    #regular = 1 if regular else -1
    simple = 1 if is_simple(X) else -1

    return convex, simple, regular


def is_convex(X):
    vector_angles = np.array([compute_theta(X[i], X[(i+1)%len(X)], X[(i+2)%len(X)]) for i in range(len(X))])

    regular = np.all(vector_angles == vector_angles[1])

    cross_product = np.transpose(vector_angles)[1]
    sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product <= 0)
        
    total_concavity = np.sum(np.transpose(vector_angles)[0])

    convex = math.isclose(total_concavity, (len(X)-1)*2*np.pi) and sign

    return convex, regular



def compute_theta(X1, X2, X3):
    d1 = X2 - X1
    d2 = X3 - X2
    v1 = np.array([np.real(d1), np.imag(d1)])
    v2 = np.array([np.real(d2), np.imag(d2)])

    if np.array_equal(v1, v2):
        return 0,0

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cross_product = np.cross(v1, v2)

    #theta = np.arcsin(cross_product / (norm_v1 * norm_v2))
    theta = np.arccos(np.dot(v1,v2)/(norm_v1 * norm_v2))


    if cross_product < 0:
        theta = np.abs(2*np.pi-theta)

    return theta, cross_product

def is_simple(X):
    n = len(X)

    if n <= 3:
            return True
        
    for i in range(n):
        p1, p2 = np.roll(X, -i)[:2]

        for j in range(i + 2, n + i - 1):
            j %= n
            q1, q2 = np.roll(X, -j)[:2]

            if intersect(p1, p2, q1, q2):
                return False

    return len(set(X)) == len(X)

def intersect(p1,p2,q1,q2):
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)

def ccw(A,B,C):
    return (np.imag(C)-np.imag(A)) * (np.real(B)-np.real(A)) > (np.imag(B)-np.imag(A)) * (np.real(C)-np.real(A))

#I want to know how the following vector has become the following equation. What transformation occurred. V = [1,w,w^2,..,w^(n-1)] where w = exp(i2pi/n) transformed to (2pi/2n)*(exp(i2pik/n)) where k=

plot(5, [[1 ,1]], 3)

plt.show()















def update_n_and_c(event, n, c, text_boxes, x, y, coefficients, coordinate_type, fig, ax):
    n = int(event)

    for i, text_box in enumerate(text_boxes):
        if i >= n:
            text_box[0].set_visible(False)
            text_box[1].set_visible(False)
        else:
            text_box[0].set_visible(True)
            text_box[1].set_visible(True)

    update_coefficients(text_boxes, coefficients, c, coordinate_type)
    
    convex_part, simple_part = complexity(x, y, n, coefficients, c, coordinate_type)
    ax.contourf(x, y, convex_part, alpha=0.5, cmap='Reds')
    ax.contourf(x, y, simple_part, alpha=0.5, cmap='Blues')
    fig.canvas.draw_idle()

def update_coefficients(text_boxes, coefficients, c, coordinate_type):
    for i, text_box in enumerate(text_boxes):
        if i != c:
            coefficients[c] = text_box[0].text + text_box[1].text*1j if coordinate_type == 'c' else text_box[0].text * np.exp(1j* text_box[1].text)

