import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, ArrowStyle
from matplotlib.collections import PatchCollection
from matplotlib import patches
import collections

'''
Code obtained from github
'''

colors = np.array(["b", "g", "r", "c", "m", "y", "k", "w"])


# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(a, x):
    r2 = np.linalg.norm(a, axis=-1, keepdims=True)**2 - (1.0)
    return r2 / np.linalg.norm(x - a, axis=-1, keepdims=True)**2 * (x - a) + a

# Inversion taking mu to origin
def reflect_at_zero(mu, x):
    a = mu / np.linalg.norm(mu, axis=-1, keepdims=True)**2
    return isometric_transform(a, x)

def hyperbolic_setup(fig, ax):
    fig.set_size_inches(10.0, 10.0, forward=True)

    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw Poincare disk boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)

# collinearity check. if collinear, draw a line and don't attempt curve
def collinear(a,b,c, eps=1e-4):

    slope1 = (c[:,1]-b[:,1])/(c[:,0]-b[:,0])
    slope2 = (a[:,1]-c[:,1])/(a[:,0]-c[:,0])
    
    return np.logical_or(np.logical_and( np.abs(c[:,0] - b[:,0]) < eps, np.abs(c[:,0]-a[:,0]) < eps ),
                        np.abs(slope1 - slope2) < eps)


def get_circle_center(a,b,c):
    m = np.zeros([len(a), 2,2])
    m[:,0,0] = 2*(c[:,0]-a[:,0])
    m[:,0,1] = 2*(c[:,1]-a[:,1])
    m[:,1,0] = 2*(c[:,0]-b[:,0])
    m[:,1,1] = 2*(c[:,1]-b[:,1])

    v = np.zeros([len(a), 2,1])
    v[:,0] = c[:,:1]**2 + c[:,1:]**2 - a[:,:1]**2 - a[:,1:]**2
    v[:,1] = c[:,:1]**2 + c[:,1:]**2 - b[:,:1]**2 - b[:,1:]**2

    return np.array([(np.linalg.inv(m_).dot(v_)).flatten() for m_, v_ in zip(m, v)])


# distance for Euclidean coordinates
def euclid_dist(a,b):
    return np.linalg.norm(a-b, axis=-1, keepdims=False)

def get_third_point(a,b):
    b0 = reflect_at_zero(a, b)
    c0 = b0 / 2.0
    c = reflect_at_zero(a, c0)

    return c

# angles for arc
def get_angles(cent, a):
    theta = np.rad2deg(np.arctan((a[:,1]-cent[:,1])/(a[:,0]-cent[:,0])))    

    quad_3_mask = np.logical_and(a[:,0]-cent[:,0] < 0, a[:,1]-cent[:,1] < 0)
    quad_2_mask = np.logical_and(a[:,0]-cent[:,0] < 0, a[:,1]-cent[:,1] >= 0)

    theta[quad_3_mask] += 180
    theta[quad_2_mask] -= 180

    theta[theta < 0] += 360
    
    theta[np.logical_and(abs(a[:,0] - cent[:,0]) < 0.1**3,  a[:,1] > cent[:,1] )] = 90
    theta[np.logical_and(abs(a[:,0] - cent[:,0]) < 0.1**3,  a[:,1] <= cent[:,1] )] = 270
    
    
    return theta

def draw_geodesic(a, b, c, ax, c1=None, c2=None, verbose=False, width=.1):
   
    cent = get_circle_center(a,b,c)  
    radius = euclid_dist(a, cent)
    t1 = get_angles(cent, b)
    t2 = get_angles(cent, a)

    mask = np.logical_or(np.logical_and(t2 > t1, t2 - t1 < 180), np.logical_and(t1 > t2, t1 - t2 >= 180))

    theta1 = np.where(mask, t1, t2)
    theta2 = np.where(mask, t2, t1)
    
    collinear_mask = collinear(a, b, c)
    mask_ = np.logical_or(collinear_mask, np.abs(t1 - t2) < 10)
    
    coordsA = "data"
    coordsB = "data"

    for ma_, a_, b_, c_, cent_, radius_, theta1_, theta2_ in zip(mask_, a, b, c, cent, radius, theta1, theta2):
        if np.random.rand() > .05:
            continue
        if ma_:
        # if True:
            e = patches.ConnectionPatch(a_, b_, coordsA, coordsB, 
                linewidth=0.05, 
                zorder=0, 
                # arrowstyle="Simple,tail_width=.01,head_width=.6,head_length=1.", 
                fc="k"
                )
            # e = patches.FancyArrowPatch(a_, b_, arrowstyle="Simple,tail_width=0.05,head_width=4,head_length=8", 
            #     color="k", zorder=0, alpha=.4, fc="r")
        else:
            e = patches.Arc((cent_[0], cent_[1]), 2*radius_, 2*radius_,
                             theta1=theta1_, theta2=theta2_, linewidth=width, fill=False, zorder=0)
            # armA = np.linalg.norm(a_ - c_)
            # c_prime = c_ - a_
            # a_prime = np.array([c_prime[0], 0])
            # angleA = np.arccos( (c_prime).dot(a_prime) / (np.linalg.norm(c_prime) * np.linalg.norm(a_prime)) )
            # armB = np.linalg.norm(b_ - c_)
            # c_prime = c_ - b_
            # b_prime = np.array([c_prime[0], 0])
            # angleB = np.arccos( (c_prime).dot(b_prime) / (np.linalg.norm(c_prime) * np.linalg.norm(b_prime)) )
            # e = patches.FancyArrowPatch(a_, b_, 
            #     arrowstyle="Simple,tail_width=.01,head_width=.6,head_length=1.", 
            #     # arrowstyle="CurveB,head_length=0.4,head_width=0.2",
            #     # arrowstyle=ArrowStyle("->", head_length=0.4, head_width=0.2),
            #     color="k", 
            #     # connectionstyle="arc,armA={},angleA={}".format(armA, np.rad2deg(angleA)), 
            #     connectionstyle="arc,armA={},angleA={},armB={},angleB={}".format(armA, np.rad2deg(angleA), armB, np.rad2deg(angleB)), 
            #     zorder=0, 
            #     alpha=.4, 
            #     fc="k")
        ax.add_patch(e)

def draw_graph(edges, embedding, labels, path, ):
    assert embedding.shape[1] == 2 

    if not isinstance(edges, np.ndarray):
        edges = np.array(edges)

    print ("saving two-dimensional poincare plot to {}".format(path))

    fig = plt.figure()
    title = "Two dimensional poincare plot"
    plt.suptitle(title)
    
    ax = fig.add_subplot(111)

    hyperbolic_setup(fig, ax)

    a = embedding[edges[:,0]]
    b = embedding[edges[:,1]]
    c = get_third_point(a, b)
    
    draw_geodesic(a, b, c, ax)
    ax.scatter(embedding[:,0], embedding[:,1], c=labels, s=50, zorder=2)

    plt.savefig(path)
    plt.close()

def plot_degree_dist(title, graph, ax):
    degrees = sorted(graph.degree().values())
    deg, counts = zip(*collections.Counter(degrees).items())
    deg = np.array(deg)
    counts = np.array(counts)

    m, c = np.polyfit(np.log(deg), np.log(counts), 1)
    y_fit = np.exp(m*np.log(deg) + c)

    ax.scatter(deg, counts, marker="x")
    ax.plot(deg, y_fit, ':', c="r")
    ax.set(title=title, xscale="log", yscale="log", xlabel="Connections", ylabel="Frequency",)
    ax.set_ylim(bottom=.9)
