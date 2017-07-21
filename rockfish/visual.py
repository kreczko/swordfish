#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.interpolate as interp
import scipy.integrate as integrate
from scipy.spatial import cKDTree, Voronoi, voronoi_plot_2d
from tqdm import tqdm

def eigen(M):
    """Calculate major and minor eigenvector and eigenvalues."""
    trM = (M[0,0]+M[1,1])
    I = np.diag([1,1])
    D = M - 0.5*I*trM
    alpha = D[0,0]
    beta = D[0,1]
    e_1 = np.array([beta, -alpha + np.sqrt(alpha**2+beta**2)])
    e_2 = np.array([beta, -alpha - np.sqrt(alpha**2+beta**2)])
    lambda_1 = 0.5*trM+np.sqrt(alpha**2+beta**2)
    lambda_2 = 0.5*trM-np.sqrt(alpha**2+beta**2)
    e_1 /= np.sqrt((e_1*e_1).sum())
    e_2 /= np.sqrt((e_2*e_2).sum())
    if e_2[0] < 0:  # Define orientation
        e_2 *= -1
    return e_1, e_2, lambda_1, lambda_2

def generate_maps(x, y, G):
    """Generate discrete versions of input metric function.

    Parameters
    ----------
        g: input metric

    Returns
    -------
        X, Y: Grid coordinates
        L1, U1, V1: Major eigenvalue and eigenvector
        L2, U2, V2: Minor eigenvalue and eigenvector
    """
    n1, n2 = np.shape(G)[:2]

    U1 = np.zeros([n1, n2])
    V1 = np.zeros([n1, n2])
    U2 = np.zeros([n1, n2])
    V2 = np.zeros([n1, n2])
    L1 = np.zeros([n1, n2])
    L2 = np.zeros([n1, n2])

    for i in range(n1):
        for j in range(n2):
            M = G[i,j]
            e_1, e_2, lambda_1, lambda_2 = eigen(M)
            U1[i,j], V1[i,j] = e_1
            U2[i,j], V2[i,j] = e_2
            L1[i,j] = lambda_1
            L2[i,j] = lambda_2

    return L1, U1, V1, L2, U2, V2

def sample(P, sample_mask = None, extent = None, wmax = 1, N = 100):
    """Return sample using rejection sampling.

    Parameters
    ----------
        P: Input density function
        wmax: parameter for rejection sampling (mass of P)
    """
    X = np.array([])
    Y = np.array([])
    xmin, xmax, ymin, ymax = extent
    P0 = lambda x, y: P(x,y)*0.9 + 0.1*wmax
    while True:
        w = np.random.random(N/2)*wmax
        x = np.random.uniform(xmin, xmax, N/2)
        y = np.random.uniform(ymin, ymax, N/2)
        accepted = P0(x,y) > w
        if sample_mask:
            accepted = accepted & sample_mask(x, y)
        X = np.append(X, x[accepted])
        Y = np.append(Y, y[accepted])
        if len(X) >= N:
            break
    sample = np.array(zip(X,Y))
    return sample


def distance(x_center, x_ball, M):
    """Return distance.

    Parameters
    ----------
        x_center: central point
        x_ball: list of points
        M: metric
    """
    dist = []
    for x in x_ball:
        dx = x_center-x
        dist.append((M.dot(dx)*dx).sum()**0.5)
    return np.array(dist)

def fisherplot(X, Y, G, streamlines = True, voronoi = False, ellipses = False,
        streamplot = False, xlog = False, ylog = False, sample_mask = None):

    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()

    # Get metric
    l1, vx1, vy1, l2, vx2, vy2 = generate_maps(X, Y, G)  # l1 > l2
    x2dim, y2dim = np.meshgrid(X, Y)
    d = np.sqrt(l1*l2)  # Density (Jeffreys' prior), sqrt(det(I))
    dmax = 1.3/np.sqrt(l2)  # Larger distance correspondingt to smaller eigenvalue
    dmin = 0.5/np.sqrt(l1)  # Smaller distance correspondingt to larger eigenvalue

    G00_interp = interp.RectBivariateSpline(X, Y, G[:,:,0,0])
    G11_interp = interp.RectBivariateSpline(X, Y, G[:,:,1,1])
    G01_interp = interp.RectBivariateSpline(X, Y, G[:,:,0,1])
    G10_interp = interp.RectBivariateSpline(X, Y, G[:,:,1,0])

    def g(x):
        return np.array(
                [[G00_interp.ev(x[0], x[1]), G01_interp.ev(x[0], x[1])],
                    [G10_interp.ev(x[0], x[1]), G11_interp.ev(x[0], x[1])]])

    vx1_interp = interp.RectBivariateSpline(X, Y, vx1/l1**0.5)
    vx2_interp = interp.RectBivariateSpline(X, Y, vx2/l2**0.5)
    vy1_interp = interp.RectBivariateSpline(X, Y, vy1/l1**0.5)
    vy2_interp = interp.RectBivariateSpline(X, Y, vy2/l2**0.5)
    d_interp = interp.RectBivariateSpline(X, Y, d)
    dmax_interp = interp.RectBivariateSpline(X, Y, dmax)
    dmin_interp = interp.RectBivariateSpline(X, Y, dmin)

    v1_fun = lambda x, t: [vx1_interp.ev(x[0], x[1]), vy1_interp.ev(x[0], x[1])]
    v2_fun = lambda x, t: [vx2_interp.ev(x[0], x[1]), vy2_interp.ev(x[0], x[1])]
    v1r_fun = lambda x, t: [-vx1_interp.ev(x[0], x[1]), -vy1_interp.ev(x[0], x[1])]
    v2r_fun = lambda x, t: [-vx2_interp.ev(x[0], x[1]), -vy2_interp.ev(x[0], x[1])]
    d_fun = lambda x, y: d_interp.ev(x, y)
    dmax_fun = lambda x, y: dmax_interp.ev(x, y)
    dmin_fun = lambda x, y: dmin_interp.ev(x, y)

    samples = sample(d_fun, sample_mask = sample_mask, extent = [xmin, xmax,
        ymin, ymax], wmax = d.max(), N = 10000)
    tree = cKDTree(samples)
    #plt.scatter(samples[:,0], samples[:,1], marker='.', alpha='0.1')
    mask = np.ones(len(samples), dtype='bool')
    for i in tqdm(range(len(samples))):
        if not mask[i]:  # Not considered anymore
            continue
        indices = tree.query_ball_point(samples[i], dmin_fun(samples[i][0],
            samples[i][1]))
        mask[indices] = False
        mask[i] = True
    samples = samples[mask]
    tree = cKDTree(samples)
    #plt.scatter(samples[:,0], samples[:,1], marker='o')

    mask = np.zeros(len(samples), dtype='bool')

    if ellipses or voronoi or streamlines:
        for i in tqdm(range(len(mask)), desc = 'Distance check'):
            indices = tree.query_ball_point(samples[i], dmax_fun(samples[i][0],
                samples[i][1]))
            dist1 = distance(samples[i], samples[indices], g(samples[i]))
            dist2 = np.array([distance(samples[i], [samples[j]], g(samples[j])) for j in
                    indices]).T[0]
            dist = np.where(dist1 < dist2, dist1, dist2)
            if not any(mask[indices] & (dist<0.7)):
                mask[i] = True

    if voronoi:
        assert not xlog and not ylog
        vor = Voronoi(samples[mask])
        voronoi_plot_2d(vor, show_vertices = False, show_points = False,
                line_colors = 'k', line_width = 1, line_alpha = 0.5)

    if ellipses:
        assert not xlog and not ylog
        for x, y in samples[mask]:
            e_1, e_2, l_1, l_2 = eigen(g([x, y]))
            ang = np.degrees(np.arccos(e_1[0]))
            e = Ellipse(xy=(x,y), width = 1/l_1**0.5, height = 1/l_2**0.5,
                    ec='0.1', fc = '0.5', angle = ang)
            plt.gca().add_patch(e)

    if streamplot:
        assert not xlog and not ylog
        plt.streamplot(x2dim, y2dim, vx1, vy1, color = 'k', linewidth = 1, density = 1, arrowstyle='-', cmap = 'cubehelix')
        plt.streamplot(x2dim, y2dim, vx2, vy2, color = 'k', linewidth = 1, density = 1, arrowstyle='-', cmap = 'cubehelix')

    if streamlines:
        from matplotlib.collections import LineCollection
        for p0 in tqdm(samples[mask], desc = 'Generate streamlines'):
            t = np.linspace(0, 0.5, 10)
            colors = ['b', 'b', 'b', 'b']
            for vfun, c in zip([v1_fun, v2_fun, v1r_fun, v2r_fun], colors):
                streamline=integrate.odeint(vfun,p0,t, rtol=1e-3, atol=1e-3)
                if xlog:
                    streamline[:,0] = 10**streamline[:,0]
                if ylog:
                    streamline[:,1] = 10**streamline[:,1]
                #plt.plot(streamline[:,0], streamline[:,1], ls='-', color='r',
                x = np.linspace(0, 1, 10)
                lwidths= np.cos(np.pi*x/2)*1+0.5
                points = streamline.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linewidths=lwidths, color=c)
                plt.gca().add_collection(lc)

    if xlog:
        plt.gca().set_xscale('log')
    if ylog:
        plt.gca().set_yscale('log')


def test():
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 2, 13)
    G = np.zeros((10, 13, 2, 2))
    g = np.array([[2., .1], [0.1, 1.]])
    G[:,:,:,:] = g
    G = (G.T*10**x).T
    sample_mask = lambda x, y: x < 2
    fisherplot(x, y, G, xlog = True, ylog = True, sample_mask = sample_mask)
    plt.savefig('ellipses.pdf')

if __name__ == "__main__":
    test()
