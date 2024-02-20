#%%
import time

# get the start time
st = time.process_time()
from vectorConvexity import PolygonAnalyzer2d
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

convexity_counter = False
edge_shapes = []

def on_click(event, coefficients, c):
    if event.inaxes is not None:
        coefficients[c] = event.x + event.y*1j
        print(event.xdata, event.ydata)
        shape = PolygonAnalyzer2d(coefficients, input_type='coefficients')
        shape.visualize()
        #plt.show()

        #flow_shape = PolygonAnalyzer2d(flow(shape.centered_X, 100), input_type='coefficients')
        #flow_shape.visualize("flow")
        plt.show()

def flow(X, t):
    n = len(X)
    w = np.exp(2*np.pi*1j*(1/n))
    j, k = np.meshgrid(range(n), range(n))
    M = (1 / np.sqrt(n))* np.matrix(np.power(w, j*k))
    #eigenvalues = np.array([np.sum(np.fromiter((coeff[j] * M[k].T[j] for j in range(len(coeff))), complex)) for k in range(len(coeff))])
    a = np.array([np.dot(X, np.asarray(M[k].getH()).ravel()) / np.dot(np.asarray(M[k]).ravel(), np.asarray(M[k].getH()).ravel()) for k in range(len(X))])
    eigenvalues = np.array([np.sum(np.fromiter((a[j] * (M*np.sqrt(len(a)))[k].T[j] for j in range(len(a))), complex)) for k in range(len(a))])
    
    return np.exp(-eigenvalues[1] * t) * np.multiply(a, eigenvalues)  

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def plot(n, c, limit):
    coeff_range = np.linspace(-limit, limit,200)
    x, y = np.meshgrid(coeff_range, coeff_range)

    convex_part, simple_part, distances = complexity(x, y, n, c)
    #regular_part

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    contours_convex = ax.contourf(x, y, convex_part, zdir='z', offset=0, alpha=0.5, cmap='Reds')
    contours_simple = ax.contourf(x, y, simple_part, zdir='z', offset=0, alpha=0.5, cmap='Blues')
    #contours_regular = ax.contourf(x, y, regular_part, alpha=0.5, cmap='Oranges')

    ax.plot_surface(x, y, 100*distances, cmap='plasma', alpha=0.8)



    #cbar = fig.colorbar(contours_convex, ax=ax)
    #cbar = fig.colorbar(contours_simple, ax=ax)
    #cbar = fig.colorbar(contours_regular, ax=ax)
    #cbar.set_label('Colorbar')

    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')

    #ax.set_xlim([-limit, limit])
    #ax.set_ylim([-limit, limit])
    
    ax.view_init(elev=30, azim=120)
    #ax.draggable(True)
    #ax.set_box_aspect([0.1,0.1,0.08])

    plt.axis('scaled')

    coefficients = np.zeros(n, dtype='complex')
    coefficients[1] = 1 + 0j
   # X = np.fft.fft(coefficients, norm='ortho')
   # w = np.exp(2*np.pi*1j*(1/n))
   # j, k = np.meshgrid(range(n), range(n))
   # M = (1 / np.sqrt(n))* np.matrix(np.power(w, j*k))
    # eigenvalues = np.array([np.sum(np.fromiter((coefficients[j] * M[k].T[j] for j in range(len(coefficients))), complex)) for k in range(len(coefficients))])
   # a = np.array([np.dot(X, np.asarray(M[k]).ravel()) / np.dot(np.asarray(M[k]).ravel(), np.asarray(M[k]).ravel()) for k in range(len(X))])
   # eigenvalues = np.array([np.sum(np.fromiter((a[j] * (M*np.sqrt(len(a)))[k].T[j] for j in range(len(a))), complex)) for k in range(len(a))])
        
    
    ##plt.scatter(eigenvalues.real, eigenvalues.imag)

    def on_scroll(event):
        ax.set_box_aspect([1,1, event.step * 0.1]) 

    #on_click_partial = partial(on_click, coefficients = coefficients, c=c)
    #on_scroll_partial = partial(on_scroll, ax=ax)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    #fig.canvas.mpl_connect('button_press_event', on_click_partial)
    #fig.savefig('filename.eps', format='eps')
    scale = 1.5
    f = zoom_factory(ax, base_scale = scale)
    plt.show()

def complexity(x, y, n, c):
    if np.isscalar(x) and np.isscalar(y):
        return compute_complexity(x, y, n, c)
    
    result_convex = np.zeros_like(x)
    result_simple = np.zeros_like(x)
    result_regular = np.zeros_like(x)
    result_distance = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result_convex[i,j], result_simple[i,j], result_regular[i,j], result_distance[i,j] = compute_complexity(x[i, j], y[i, j], n, c)
    
    return result_convex, result_simple, result_distance

def compute_complexity(x, y, n, c):
    coefficients = np.zeros(n, dtype='complex')
    coefficients[1] = 1 + 0j
    coefficients[c] = x + y*1j
    #coefficients[int(np.ceil(1.6*c))] = 0+y * 1j
    #coefficients[int(np.ceil(2.1*c))] = x + 0j
    X = np.fft.fft(coefficients, norm='ortho')

    convex, regular = is_convex(X)
    simple = is_simple(X)


    normalized_regular = np.exp(1j * np.linspace(0, 2 * np.pi, len(X) + 1))[:-1]
    normalized_X = X / np.abs(X)

    distances = np.abs(normalized_X[:, np.newaxis] - normalized_regular)

    # Find the minimum distance for each vertex in `X`
    min_distances = np.min(distances, axis=1)

    # Calculate the maximum distance as the Hausdorff distance
    sum_of_distances = np.max(min_distances)

    # Calculate the distances between every pair of vertices
    #distances = np.linalg.norm(normalized_X - np.roll(normalized_X, 1))

    # Compute the sum of distances
    #sum_of_distances = np.sum(distances)


    #evolved_polygon = PolygonAnalyzer2d(flow(X, 100), input_type='coefficients')
    #regular = 1 if evolved_polygon.total_angle == 0 else -1

    #global convexity_counter
    #if convex != convexity_counter:
    #    edge_shapes.append((X, convex))
    #    convexity_counter = convex

    #sum_of_distances = 2*sum_of_distances if convex else sum_of_distances/2
    #sum_of_distances = sum_of_distances if simple else 0

    convex = 1 if convex else -1
    #regular = 1 if regular else -1
    simple = 1 if simple else -1

    return convex, simple, regular, sum_of_distances


def is_convex(X):
    vector_angles = np.array([compute_theta(X[i], X[(i+1)%len(X)], X[(i+2)%len(X)]) for i in range(len(X))])

    regular = np.all(vector_angles == vector_angles[1])

    cross_product = np.transpose(vector_angles)[1]
    sign = np.all(cross_product > 0) if cross_product[0] > 0 else np.all(cross_product <= 0)
        
    total_concavity = np.sum(np.transpose(vector_angles)[0])

    convex = (total_concavity <= 2*np.pi or math.isclose(total_concavity, 2*np.pi)) and sign

    return convex, regular #np.sum(vector_angles)/len(X)


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



plot(5, 2, 1.7)
et = time.process_time()

# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')
#case1 = PolygonAnalyzer2d(edge_shapes[10][0])
#print(edge_shapes[10][1])
#print(case1.coefficients)
#case1.visualize()
plt.show()
# %%


























"""def is_convex2(polygon):
    arguments = np.angle(polygon)
    arguments = np.mod(arguments, 2 * np.pi)

    diff_to_2pi = 2 * np.pi - arguments[0]
    arguments += diff_to_2pi
    arguments = np.mod(arguments, 2 * np.pi)
    arguments[np.isclose(arguments, 0)] = 2*np.pi

    differences = np.diff(arguments)
    is_ascending = np.all(differences >= 0)
    is_descending = np.all(differences <= 0)

    return is_ascending or is_descending

def onSegment(p, q, r):
    if ( (q.real <= max(p.real, r.real)) and (q.real >= min(p.real, r.real)) and 
           (q.imag <= max(p.imag, r.imag)) and (q.imag >= min(p.imag, r.imag))):
        return True
    return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (float(q.imag - p.imag) * (r.real - q.real)) - (float(q.real - p.real) * (r.imag - q.imag))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def intersectb(p1,q1,p2,q2):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False


def intersect3(v1, v2):
    x1, y1 = v1[0].real, v1[0].imag
    x2, y2 = v1[1].real, v1[1].imag
    x3, y3 = v2[0].real, v2[0].imag
    x4, y4 = v2[1].real, v2[1].imag

    det1 = (x1 - x2) * (y3 - y4)
    det2 = (y1 - y2) * (x3 - x4)
    det3 = (x1 - x2) * (y3 - y2)
    det4 = (y1 - y2) * (x3 - x2)

    if det1 == det2 == det3 == det4 == 0:
        # Line segments are collinear
        if min(x1, x2) <= max(x3, x4) and min(x3, x4) <= max(x1, x2) and min(y1, y2) <= max(y3, y4) and min(y3, y4) <= max(y1, y2):
            return True
    elif det1 != det2:
        t1 = (x1 * (y3 - y4) + x3 * (y4 - y1) + x4 * (y1 - y3)) / (det1 - det2)
        t2 = -(x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / (det1 - det2)

        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return True

    return False


def intersect2(v1, v2):
    da = v1[1] - v1[0]
    db = v2[1] - v2[0]
    dp = v1[0] - v2[0]
    
    denom = np.dot(da, db)
    num = np.dot(da, dp)
    
    if denom == 0:
        return False
    
    t = num / denom
    return 0 <= t <= 1

def intersecta(p1, p2, q1, q2):
    a1 = np.array([p1.real, p1.imag])
    a2 = np.array([p2.real, p2.imag])
    b1 = np.array([q1.real, q1.imag])
    b2 = np.array([q2.real, q2.imag])

    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:
        return False
    return True


def is_simpl1(polygon):
    arguments = np.angle(polygon)
    arguments = np.mod(arguments, 2 * np.pi)

    arguments[np.isclose(arguments, 0)] = 2*np.pi
    differences = np.diff(arguments)
    is_ascending = np.all(differences >= 0)
    is_descending = np.all(differences <= 0)
    

    return is_ascending or is_descending

def is_simple1(polygon):
    # Compute the convex hull of the polygon's vertices
    hull = ConvexHull(polygon)
    hull_vertices = hull.vertices

    # Check if the vertices of the polygon match the vertices of the convex hull
    is_simple = np.array_equal(polygon, polygon[hull_vertices])

    return is_simple"""