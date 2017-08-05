#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy.interpolate as ip
from scipy.integrate import odeint, dblquad
import pylab as plt
from matplotlib.patches import Ellipse
from random import randint

#def eigen(g):
#    """Return major and minor eigenvector and eigenvalues.
#
#    Arguments
#    ---------
#    g : (2,2) array
#
#    Returns
#    -------
#    e1 : 1D-array (2)
#        Eigenvector
#    e2 : 1D-array (2)
#        Eigenvector
#    l1 : float
#        Eigenvalue
#    l2 : float
#        Eigenvalue
#    """
#    # FIXME: Remove?
#    w, v = np.linalg.eig(g)
#    return v[:,0], v[:,1], w[0], w[1]
#
#def eigen(M):
#    """Return major and minor eigenvector and eigenvalues.
#
#    Arguments
#    ---------
#    M : (2,2) array
#
#    Returns
#    -------
#    e1 : 1D-array (2)
#        Eigenvector
#    e2 : 1D-array (2)
#        Eigenvector
#    l1 : float
#        Eigenvalue
#    l2 : float
#        Eigenvalue
#    """
#    trM = (M[0,0]+M[1,1])
#    I = np.diag([1,1])
#    D = M - 0.5*I*trM
#    alpha = D[0,0]
#    beta = D[0,1]
#    e1 = np.array([beta, -alpha + np.sqrt(alpha**2+beta**2)])
#    e2 = np.array([beta, -alpha - np.sqrt(alpha**2+beta**2)])
#    l1 = 0.5*trM+np.sqrt(alpha**2+beta**2)
#    l2 = 0.5*trM-np.sqrt(alpha**2+beta**2)
#    e1 /= np.sqrt((e1*e1).sum())
#    e2 /= np.sqrt((e2*e2).sum())
#    # FIXME: Is this required?
#    if e2[0] < 0:  # Define orientation
#        e2 *= -1
#    return e1, e2, l1, l2

def tensor_to_vector(g):
    # Generate vector field
    _, _, N, M = np.shape(g)
    L1 = np.zeros((N,M))
    L2 = np.zeros((N,M))
    e1 = np.zeros((N,M,2))
    e2 = np.zeros((N,M,2))
    for i in range(N):
        for j in range(M):
            w, v = np.linalg.eig(g[:,:,i,j])
            e1[i,j] = v[:,0], 
            e2[i,j] = v[:,1], 
            L1[i,j] = w[0]
            L2[i,j] = w[1]

    def swap(i,j):
        e_tmp = e1[i,j]
        l_tmp = L1[i,j]
        e1[i,j] = e2[i,j]
        L1[i,j] = L2[i,j]
        e2[i,j] = e_tmp
        L2[i,j] = l_tmp

    # Reorder vector field
    for j in range(0, M):
        if j > 0:
            if abs((e1[0,j]*e1[0,j-1]).sum())<abs((e2[0,j]*e1[0,j-1]).sum()):
                swap(i,j)
        for i in range(1, N):
            #      if abs((e1[i,j]*e1[i-1,j]).sum())<abs((e2[i,j]*e1[i-1,j]).sum()):
            #         swap(i,j)
            if (e1[i,j]*e1[i-1,j]).sum() < 0:
                e1[i,j] *= -1
            if (e2[i,j]*e2[i-1,j]).sum() < 0:
                e2[i,j] *= -1

    return e1, L1, e2, L2

    #plt.quiver(X, Y, e1[:,:,0], e1[:,:,1])
    #plt.savefig('text.eps')


#tensor_to_vector(g)
#quit()

class TensorField(object):
    """Object to generate tensor field."""
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.extent = [x.min(), x.max(), y.min(), y.max()]
        self.g00 = ip.RectBivariateSpline(x, y, z[0,0])
        self.g11 = ip.RectBivariateSpline(x, y, z[1,1])
        self.g01 = ip.RectBivariateSpline(x, y, z[0,1])
        self.g10 = self.g01

    def __call__(self, x, y, dx = 0, dy = 0):
        g00 = self.g00(x, y, dx = dx, dy = dy)[0,0]
        g11 = self.g11(x, y, dx = dx, dy = dy)[0,0]
        g01 = self.g01(x, y, dx = dx, dy = dy)[0,0]
        g10 = g01
        return np.array([[g00, g01], [g10, g11]])

    def Christoffel_1st(self, x, y):
        """Return Christoffel symbols, Gamma_{abc}."""
        g  = self.__call__(x, y)
        gx = self.__call__(x, y, dx = 1)
        gy = self.__call__(x, y, dy = 1)
        G000 = 0.5*gx[0,0]
        G111 = 0.5*gy[1,1]
        G001 = 0.5*(gy[0,0]+gx[0,1]-gx[0,1])
        G011 = 0.5*(gy[0,1]+gy[0,1]-gx[1,1])
        G010 = 0.5*(gx[0,1]+gy[0,0]-gx[1,0])
        G110 = 0.5*(gx[1,1]+gy[1,0]-gy[1,0])
        G100 = 0.5*(gx[1,0]+gx[1,0]-gy[0,0])
        G101 = 0.5*(gy[1,0]+gx[1,1]-gy[0,1])
        return np.array([[[G000, G001],[G010, G011]], [[G100, G101],[G110, G111]]])

    def Christoffel_2nd(self, x, y):
        Christoffel_1st = self.Christoffel_1st(x, y)
        g = self.__call__(x, y)
        inv_g = np.linalg.inv(g)
        return g.dot(Christoffel_1st)

    def _func(self, v, t=0):
        r = np.zeros_like(v)
        G = self.Christoffel_2nd(v[0], v[1])
        r[0] = v[2]
        r[1] = v[3]
        r[2] = G[0,0,0]*v[2]*v[2]+ G[0,0,1]*v[2]*v[3]+ G[0,1,0]*v[3]*v[2]+ G[0,1,1]*v[3]*v[3]
        r[3] = G[1,0,0]*v[2]*v[2]+ G[0,0,1]*v[2]*v[3]+ G[0,1,0]*v[3]*v[2]+ G[0,1,1]*v[3]*v[3]
        return r

    def volume(self):
        density = lambda x, y: np.sqrt(np.linalg.det(self.__call__(x, y)))
        xmin, xmax, ymin, ymax = self.extent
        return dblquad(density, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]

    def sample(self, mask = None, N = 100):
        X = np.zeros(0)
        Y = np.zeros(0)
        xmin, xmax, ymin, ymax = self.extent
        wmax = 0.  # Determine wmax dynamically
        while True:
            w = np.random.random()*wmax
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            P = np.linalg.det(self.__call__(x,y))
            if wmax < P: wmax = P
            accepted = P >= w
            if mask:
                accepted = accepted & mask(x, y)
            if accepted:
                X = np.append(X, x)
                Y = np.append(Y, y)
            if len(X) >= N:
                break
        sample = np.array(zip(X,Y))
        return sample

    def relax(self, samples, spring = False, boundaries = None):
        # Using Kindlmann & Westin 2006 prescription
        glist = []
        N = len(samples)
        gamma = 0.5
        if spring:
            phi_tilde = lambda r: (
                   r-1 if r < 1 else (r-1)*(1+gamma-r)**2/gamma**2 if r < 1+gamma else 0
                   )
        else:
            phi_tilde = lambda r: r-1 if r < 1 else 0
            beta = 0.1
            phi_tilde = lambda r: r-1-beta if r < 1 else beta*(r-2) if r < 2 else 0
        for p in samples:
            glist.append(self.__call__(p[0], p[1]))
        f_list = np.zeros((N, 2))
        for i in range(N):
            for j in range(i):
                gmean = 0.5*(glist[i]+glist[j])
                x = samples[i]-samples[j]
                D = np.sqrt(x.T.dot(gmean.dot(x)))
                f = phi_tilde(D)/D*gmean.dot(x)
                f_list[i] += -f
                f_list[j] += f
        if boundaries is None:
            boundaries = self.extent
        for i in range(N):
            bxL = samples[i][0]-boundaries[0]
            bxR = samples[i][0]-boundaries[1]
            byB = samples[i][1]-boundaries[2]
            byT = samples[i][1]-boundaries[3]
            alpha = 1
            f_list[i][0] -= alpha*bxL if bxL<0 else 0
            f_list[i][0] -= alpha*bxR if bxR>0 else 0
            f_list[i][1] -= alpha*byB if byB<0 else 0
            f_list[i][1] -= alpha*byT if byT>0 else 0

        return samples + np.array(f_list)*1.0

    def ellipses(self, samples):
        for x in samples:
            e_1, e_2, l_1, l_2 = eigen(self.__call__(x[0], x[1]))
            ang = np.degrees(np.arccos(e_1[0]))
            e = Ellipse(xy=x, width = 1/l_1**0.5, height = 1/l_2**0.5,
                 ec='0.1', fc = '0.5', angle = ang)
            plt.gca().add_patch(e)

    def get_vector_field():
        raise NotImplementedError()


class VectorField(object):
    """VectorField(x, y, v, d)

    This class represents a vector field and allows variable-density streamline
    plotting.
    """
    def __init__(self, x, y, v, d):
        """Creat a VectorField object from a vector field on a grid.
  
        Arguments
        ---------
        x : 1-D array (N)
            Defines x-grid.
        y : 1-D array (N)
            Defines y-grid.
        v : 2-D array (2, N)
            Defines vector field on grid.
        d : 1-D array (N)
            Defines distance between streamlines.
        """
  
        self.x, self.y, self.v, self.d = x, y, v, d
        self.extent = [x.min(), x.max(), y.min(), y.max()]
        self.v0 = ip.RectBivariateSpline(x, y, v[0])
        self.v1 = ip.RectBivariateSpline(x, y, v[1])
        self.d = ip.RectBivariateSpline(x, y, d)
        self.lines = []

    def __call__(self, x, t=0):
        """Return interpolated vector at position x.
  
        Arguments
        ---------
        x : 1-D array (2)
        """
        return np.array([self.v0(x[0], x[1])[0,0], self.v1(x[0], x[1])[0,0]])

    def dist(self, x):
        """Return interpolated streamline distance at position x.

        Arguments
        ---------
        x : 1-D array (2)
        """
        return self.d(x[0], x[1])[0,0]

    def _boundary_mask(self, seg, boundaries):
        """Returns mask for line segments outside of the boundaries.

        Arguments
        ---------
        seg : 2-D array (2, N)
            Line segment.

        Returns
        -------
        mask : 1-D array (N)
        """
        xmin, xmax, ymin, ymax = self.extent
        mask = (seg[:,0] < xmax) & (seg[:,0] > xmin)& (seg[:,1] < ymax)& (seg[:,1] > ymin)
        if boundaries is not None:
            mask *= boundaries(seg[:,0], seg[:,1])
        return mask

    def _proximity_mask(self, seg, lines):
        """Returns mask for line segments that lie to close to other lines.

        Arguments
        ---------
        seg : 2-D array (2, N)
            Line segment.

        Returns
        -------
        mask : 1-D array (N)
        """
        mask = np.ones(len(seg), dtype='bool')
        for i, x in enumerate(seg):
            for line in lines:
                dist_min = np.sqrt(((x-line)**2).sum(axis=1)).min()
                #dist_min_major = abs(((x-line)*self.major(x)).sum(axis=1)).min()
                #dist_min_minor = abs(((x-line)*self.minor(x)).sum(axis=1)).min()
                # FIXME: Hardcoded minimal distance
                if dist_min < self.dist(x)*0.75:
                    mask[i:] = False
                    return mask
        return mask

    def _get_streamline(self, xinit, lines, boundaries):
        """Generate next streamline.

        Arguments
        ---------
        xinit : 1-D array
            Start position.
        """
        l = []
        while True:
            t = np.linspace(0, 1, 10)
            x0 = l[-1] if l != [] else xinit
            lnew = odeint(self.__call__, x0, t)
            maskb = self._boundary_mask(lnew, boundaries)
            maskp = self._proximity_mask(lnew, lines)
            if all(maskb) and all(maskp):
                l.extend(lnew)
            else:
                l.extend(lnew[maskb&maskp])
                break
        l.reverse()
        while True:
            t = np.linspace(0, -1, 10)
            x0 = l[-1] if l != [] else xinit
            lnew = odeint(self.__call__, x0, t)
            maskb = self._boundary_mask(lnew, boundaries)
            maskp = self._proximity_mask(lnew, lines)
            if all(maskb) and all(maskp):
                l.extend(lnew)
            else:
                l.extend(lnew[maskb&maskp])
                break
        line = np.array(l)
        return line

    def _seed(self, lines, Nmax = 1000, boundaries = None):
        """Generate new seed position for next streamline.
  
        Arguments
        ---------
        Nmax : integer (optional)
            Maximum number of trials, default is 1000.
        """
        for k in range(Nmax):
            j = randint(0, len(lines)-1)
            i = randint(0, len(lines[j])-1)
            x = lines[j][i]
            v = self.__call__(x)
            v_orth = np.array([v[1], -v[0]])/(v[0]**2+v[1]**2)**0.5
            xseed = x + v_orth*self.dist(x)*(-1)**randint(0,1)
            inbounds = self._boundary_mask(np.array([xseed]), boundaries)[0]
            notclose = self._proximity_mask(np.array([xseed]), lines)[0]
    #         plt.plot([x[0],xseed[0]], [x[1],xseed[1]], marker='', ls='-', color='b')
    #         plt.plot(xseed[0], xseed[1], marker='.', ls='', color='b')
            if inbounds & notclose:
                return xseed
        return None

    def get_streamlines(self, xinit, mask = None, Nmax = 100):
        """Generate streamlines.

        Arguments
        ---------
        xinit: (2) array
            Position of initial streamline.
        mask : function
            Boolean valued mask function.
        Nmax : int (optional)
            Maximum number of requested streamlines, default 30.
        """
        lines = []
        xseed = xinit
        for i in range(Nmax):
            line = self._get_streamline(xseed, lines, mask)
            lines.append(line)
            xseed = self._seed(lines, boundaries = mask)
            if xseed is None: break
        return lines

def test_vf():
    plt.figure(figsize=(4,4))

    x = np.linspace(0, 10, 40)
    y = np.linspace(0, 10, 41)
    Y, X = np.meshgrid(y, x)

    v0 = -np.ones_like(X)*1.0
    v1 = np.sin(X/4)*2

    v = np.array([v0, v1])
    d = np.sqrt(Y)*np.sqrt(X)*0.3+0.02

    vo = np.array([v1, -v0])
    do = np.ones_like(X)*0.4

    vf = VectorField(x, y, v, d)
    mask = lambda x, y: ((x-5)**2+(y-5)**2)**0.5<5
    lines = vf.get_streamlines([5,5], mask = mask)
    for line in lines:
        plt.plot(line.T[0], line.T[1], color='0.5', lw=0.5)

    vf = VectorField(x, y, vo, do)
    lines = vf.get_streamlines([5,5], mask = mask)
    for line in lines:
        plt.plot(line.T[0], line.T[1], color='0.5', lw=0.5)

    plt.xlim([-2,12])
    plt.ylim([-2,12])
    plt.savefig('test_vf.eps')

#def test2():
#    g = np.array([[2, 0.0], [0.0, 10]])
#    print eigen(g)

def test():
    x = np.linspace(0, 10, 40)
    y = np.linspace(0, 10, 41)

    Y, X = np.meshgrid(y, x)

    z00 = np.sin(X*3)*np.cos(Y*3)*0.0+0.3
    z11 = np.sin(X*3)*np.cos(Y*3)*0+0.3
    z01 = np.sin(X*3)*np.cos(Y*3)*0+0.20
    z10 = np.sin(X*3)*np.cos(Y*3)*0+0.20

    v0 = -np.ones_like(X)*1.0
    v1 = np.sin(X/4)*2
    v = np.array([v0, v1])
    vo = np.array([v1, -v0])
    d = np.sqrt(Y)*np.sqrt(X)*0.3+0.02
    do = np.ones_like(X)*0.4

    g = np.array([[z00, z01], [z10, z11]])

    #m = ip.RectBivariateSpline(x, y, z, kx=3, ky=3)

    #xf = np.linspace(0, 10, 1000)

    #plt.plot(xf, m(xf, 1))
    #plt.plot(xf, m(xf, 1, dx=1, dy=1))
    #plt.plot(x, m(x, 1), marker='o', ls='')
    #plt.show()
    plt.figure(figsize=(4,4))

    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 40)

    Y, X = np.meshgrid(y, x)

    z00 = np.ones_like(X)*1.0
    z01 = np.zeros_like(X)+0.00001
    z10 = np.zeros_like(X)+0.00001
    z11 = np.ones_like(X)+1.0

    R = np.sqrt(X**2+Y**2)

    z00 += 5*Y**2/R**2
    z01 += -5*X*Y/R**2
    z10 += -5*X*Y/R**2
    z11 += 5*X**2/R**2

    g = np.array([[z00, z01], [z10, z11]])



    vf = VectorField(x, y, v, d)
    xseed = [5,5]

    vf = VectorField(x, y, vo, do)
    xseed = [5,5]
    for i in range(30):
        line = vf.streamline(xseed)
        plt.plot(line.T[0], line.T[1], color='0.5', lw=0.5)
        xseed = vf.seed()
        if xseed is None: break
    #quit()
    #print vf.proximity_mask([[1, 1], [1,1]], [line])
    #quit()
    plt.xlim([-2,12])
    plt.ylim([-2,12])


    # x' = v
    # v' = - Gamma(x) v v

    tf = TensorField(x, y, g)

    t = np.linspace(0, 5, 10)
    y = odeint(tf._func, [3, 3, 0., 1], t)
    plt.plot(y[:,0], y[:,1], 'r')
    plt.savefig('text.eps')
    quit()

    #plt.scatter(sample[:,0], sample[:,1], marker='.')
    for i in range(20):
        print i
        sample = tf.relax(sample, spring=False)
    #for i in range(1000):
    #   print i
    #   sample = tf.relax(sample, spring=True)
       #plt.scatter(sample[:,0], sample[:,1], marker='.', zorder=100)
    tf.ellipses(sample)
    plt.xlim([-2,12])
    plt.ylim([-2,12])
    plt.savefig('text.eps')

#def test2():
#    g = np.array([[2, 0.0], [0.0, 10]])
#    print eigen(g)

if __name__ == "__main__":
    test_vf()
    #test()
