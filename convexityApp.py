#%%
from vectorConvexity import PolygonAnalyzer2d
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from functools import partial

convexity_counter = False
edge_shapes = []

def refreshView(text_boxes, x, y, n, coefficients, c, coordinate_type, fig, ax):
    plt.cla()
    for text_box in text_boxes:
        if text_box[2] == c:
            continue
        print(text_box[0], text_box[2], c)
        x_i = float(text_box[0].text)
        y_i = float(text_box[1].text)
        coefficients[text_box[2]] = x_i + y_i*1j if coordinate_type == 'c' else x_i * np.cos(y_i) + 1j * x_i * np.sin(y_i)
    convex_part, simple_part = complexity(x, y, n, coefficients, c, coordinate_type)
        
    ax.contourf(x, y, convex_part, alpha=0.5, cmap='Reds')
    ax.contourf(x, y, simple_part, alpha=0.5, cmap='Blues')
    for i in range(20):
        ax.plot((float(i+0.001)*0.1)* np.cos(np.linspace(0, 2*np.pi, 100)), (float(i+0.001)*0.1)* np.sin(np.linspace(0, 2*np.pi, 100)))
    fig.canvas.draw_idle()
    return text_boxes, coefficients

def update_n_and_c(event, changed, stable, text_boxes, x, y, coefficients, coordinate_type, fig, ax, old_c=-1):
    if changed == 'n':
        n, c = int(event), stable
    else:
        n, c = stable, int(event)

    if old_c != -1:
        ax_text_real = plt.axes([0.05, 0.9 - old_c * 0.07, 0.1, 0.03])
        ax_text_imag = plt.axes([0.18, 0.9 - old_c * 0.07, 0.1, 0.03])
        text_box_real = widgets.TextBox(ax_text_real, f'a_{old_c}: Real:', initial=str(0.0))
        text_box_imag = widgets.TextBox(ax_text_imag, 'Imag:', initial=str(0.0))
        text_boxes[old_c] = [text_box_real, text_box_imag, old_c]
        if old_c == c:
            print('alert!!!!')
        text_boxes[c] = ['100.0', '100.0',c]

        for txt in fig.texts:
            if txt._text_box.captured is not None and txt._text_box[2] == c:
                txt.set_visible(False)
                txt.remove()
            
    return refreshView(text_boxes, x, y, n, coefficients, c, coordinate_type, fig, ax)

def update_n(event, c, text_boxes, x, y, coefficients, coordinate_type, fig, ax):
    n = int(event)
    for txt in fig.texts:
        txt.remove()

    text_boxes = []
    for i in range(n):
        if i == c:
            text_boxes.append(['100.0', '100.0', i])
            continue
        ax_text_real = plt.axes([0.05, 0.9 - i * 0.07, 0.1, 0.03])
        ax_text_imag = plt.axes([0.18, 0.9 - i * 0.07, 0.1, 0.03])
        text_box_real = widgets.TextBox(ax_text_real, f'a_{i}: Real:', initial=str(0.0))
        text_box_imag = widgets.TextBox(ax_text_imag, 'Imag:', initial=str(0.0))

        text_boxes.append([text_box_real, text_box_imag, i])

    coefficients = np.zeros(n, dtype='complex')

    return update_n_and_c(n, changed='n', stable=c, text_boxes=text_boxes, x=x, y=y, coefficients=coefficients,
                                     coordinate_type=coordinate_type, fig=fig, ax=ax)

def on_click(event, coefficients, c, coordinate_type):
    if event.inaxes is not None and event.key == 'control':
        coefficients[c] = event.xdata + event.ydata*1j #if coordinate_type == 'c' else event.xdata * np.cos(event.ydata) + 1j * event.xdata * np.sin(event.ydata)
        shape = PolygonAnalyzer2d(coefficients, input_type='coefficients')
        shape.visualize()
        plt.show()

def on_submit(event, text_boxes, x, y, n, coefficients, c, coordinate_type, fig, ax):
    if event.key == 'shift':
        refreshView(text_boxes, x, y, n, coefficients, c, coordinate_type, fig, ax)

def plot(n=5, o=[[2,1]], c=2, coordinate_type='c', resolution=100, limit =5):
    if coordinate_type == 'c':
        coeff_range = np.linspace(-limit,limit,resolution)
        x, y = np.meshgrid(coeff_range, coeff_range)
    else:
        coeff_range1 = np.linspace(-limit,limit,resolution)
        coeff_range2 = np.linspace(-limit,limit,resolution)
        #coeff_range2 = np.linspace(0, 2*np.pi,resolution)
        x, y = np.meshgrid(coeff_range1, coeff_range2)
        

    coefficients = np.zeros(n, dtype='complex')
    for pair in o:
        coefficients[pair[0]] = pair[1]
    X = PolygonAnalyzer2d(coefficients, input_type='coeff')

    convex_part, simple_part = complexity(x, y, n, coefficients, c, coordinate_type)

    fig, ax = plt.subplots()
    ax.contourf(x, y, convex_part, alpha=0.5, cmap='Reds')
    ax.contourf(x, y, simple_part, alpha=0.5, cmap='Blues')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')

    plt.axis('scaled')
    
    text_boxes = []
    for i, coeff in enumerate(coefficients):
        if i == c:
            text_boxes.append(['100.0', '100.0', i])
            continue
        ax_text_real = plt.axes([0.05, 0.9 - i * 0.07, 0.1, 0.03])
        ax_text_imag = plt.axes([0.18, 0.9 - i * 0.07, 0.1, 0.03])
        text_box_real = widgets.TextBox(ax_text_real, f'a_{i}: Real:', initial=str(np.real(coeff)))
        text_box_imag = widgets.TextBox(ax_text_imag, 'Imag:', initial=str(np.imag(coeff)))

        text_boxes.append([text_box_real, text_box_imag, i])

    def update_n_partial(event):
        nonlocal n, text_boxes, coefficients
        text_boxes, coefficients = update_n(event, c=c, text_boxes=text_boxes, x=x, y=y, coefficients=coefficients,
                                     coordinate_type=coordinate_type, fig=fig, ax=ax)
        n = int(event)
    text_n = widgets.TextBox(plt.axes([0.8, 0.06, 0.1, 0.03]), 'n:', initial=str(n))

    text_n.on_submit(update_n_partial)

    def update_c_partial(event):
        nonlocal c, text_boxes, coefficients
        text_boxes, coefficients = update_n_and_c(event, changed='c', stable=n, text_boxes=text_boxes, x=x, y=y, coefficients=coefficients,
                       coordinate_type=coordinate_type, fig=fig, ax=ax, old_c=c)
        c = int(event)
    text_c = widgets.TextBox(plt.axes([0.8, 0.02, 0.1, 0.03]), 'c:', initial=str(c))
    text_c.on_submit(update_c_partial)

    on_submit_partial = lambda event: on_submit(event, text_boxes=text_boxes, x=x, y=y, n=n, coefficients=coefficients, c=c,
                                                coordinate_type=coordinate_type, fig=fig, ax=ax)
    fig.canvas.mpl_connect('key_release_event', on_submit_partial)

    on_click_partial = lambda event: on_click(event, coefficients = coefficients, c=c, coordinate_type=coordinate_type)
    fig.canvas.mpl_connect('button_press_event', on_click_partial)

    fig.savefig('filename.eps', format='eps')
    plt.show()

def complexity(x, y, n, coefficients, c, coordinate_type):
    if np.isscalar(x) and np.isscalar(y):
        return compute_complexity(x, y, n, coefficients, c, coordinate_type)
    
    result_convex = np.zeros_like(x)
    result_simple = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result_convex[i,j], result_simple[i,j] = compute_complexity(x[i, j], y[i, j], n, coefficients, c, coordinate_type)
    
    return result_convex, result_simple

def compute_complexity(x, y, n, coefficients, c, coordinate_type):
    coefficients[c] = x + y*1j #if coordinate_type == 'c' else x * np.cos(y) + 1j * x * np.sin(y)
    #coefficients[c+1] = (1/2)*(np.sqrt(x + y*1j)) - 0.2 + 0.08*1j
    X = np.fft.fft(coefficients, norm='ortho')

    convex = is_convex(X)

    global convexity_counter
    if convex != convexity_counter:
        edge_shapes.append((X, convex))
        convexity_counter = convex

    convex = 1 if convex else -1
    simple = 1 if is_simple(X) else -1

    return convex, simple

def is_convex(X):
    vector_angles = np.array([compute_theta(X[i], X[(i+1)%len(X)], X[(i+2)%len(X)]) for i in range(len(X))])

    cross_product = np.transpose(vector_angles)[1]
    sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product <= 0)
        
    total_concavity = np.sum(np.transpose(vector_angles)[0])

    convex = math.isclose(total_concavity, (len(X)-1)*2*np.pi) and sign

    return convex

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

plot(coordinate_type='p', n=5, limit=2, c=3, o = [[1, 1]])
# %%