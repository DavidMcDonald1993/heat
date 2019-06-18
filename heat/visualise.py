import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, ArrowStyle
from matplotlib.collections import PatchCollection
from matplotlib import patches
import collections
import networkx as nx

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

def draw_geodesic(a, b, c, ax, c1=None, c2=None, verbose=False, width=.05):
   
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
        if ma_:
            e = patches.ConnectionPatch(a_, b_, coordsA, coordsB, 
                linewidth=width, 
                zorder=0, 
                )
        else:
            e = patches.Arc((cent_[0], cent_[1]), 2*radius_, 2*radius_,
                             theta1=theta1_, theta2=theta2_, linewidth=width, fill=False, zorder=0)
        ax.add_patch(e)

def draw_graph(graph, embedding, labels, path, s=25):
    assert embedding.shape[1] == 2 

    edges = list(graph.edges())

    if labels is not None:
        # filter out noise nodes labelled as -1
        idx, = np.where(labels[:,0] > -1)
        num_labels = int(max(set(labels[:,0])) + 1)
        # colours = np.random.rand(num_labels, 3)
        colours = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [0,0,0],
            [1,1,1]
            ]) 
        # colours = np.array(["r", "g", "b", "y", "m", "c"])
        assert num_labels < len(colours)
    else:
        idx = np.arange(len(embedding))

    if not isinstance(edges, np.ndarray):
        edges = np.array(edges)

    print ("saving two-dimensional poincare plot to {}".format(path))

    fig = plt.figure()
    title = "Two dimensional poincare plot"
    plt.suptitle(title)
    
    ax = fig.add_subplot(111)

    hyperbolic_setup(fig, ax)

    # a = embedding[edges[:,0]]
    # b = embedding[edges[:,1]]
    # c = get_third_point(a, b)
    
    # draw_geodesic(a, b, c, ax)

    # # s = {n: (bc+.05) * 100 for n, bc in nx.betweenness_centrality(graph).items()}
    # # s = [s[n] for n in sorted(graph.nodes)]
    # s = np.array([graph.degree(n, weight="weight") for n in sorted(graph.nodes())])
    # s = s / s.max() * 100
    # ax.scatter(embedding[idx,0], embedding[idx,1], 
    #     c=colours[labels[idx,0]] if labels is not None else None,
    #     s=s, zorder=2)

    pos = {n: emb for n, emb in zip(sorted(graph.nodes()), embedding)}
    node_colours = np.array([colours[labels[n, 0]] for n in graph.nodes()]) if labels is not None else None
    
    # bc  = nx.betweenness_centrality(graph)
    # node_sizes = np.array([(bc[n] + .05) * 50 for n in sorted(graph.nodes())])
    node_sizes = np.array([graph.degree(n, weight="weight") for n in graph.nodes()])
    node_sizes = node_sizes / node_sizes.max() * 250
    nx.draw_networkx_nodes(graph, pos=pos, node_color=node_colours, node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos=pos, width=.05, node_size=node_sizes)
    # nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=nx.get_edge_attributes(graph, name="weight"))

    plt.savefig(path)
    plt.close()

def plot_degree_dist(graph, title):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    degrees = sorted(dict(graph.degree(weight="weight")).values())

    deg, counts = zip(*collections.Counter(degrees).items())
    deg = np.array(deg)
    counts = np.array(counts)

    idx = deg > 0
    deg = deg[idx]
    counts = counts[idx]

    m, c = np.polyfit(np.log(deg), np.log(counts), 1)
    y_fit = np.exp(m*np.log(deg) + c)

    ax.scatter(deg, counts, marker="x")
    ax.plot(deg, y_fit, ':', c="r")
    ax.set(title=title, xscale="log", yscale="log", xlabel="Connections", ylabel="Frequency",)
    ax.set_ylim(bottom=.9)
    plt.show()
