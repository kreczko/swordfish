#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy.interpolate as ip
from scipy.integrate import odeint, dblquad
import pylab as plt
from matplotlib.patches import Ellipse
from random import randint

class TensorField(object):
    """Object to generate tensor field."""
    def __init__(self, x, y, g, logx = False, logy = False):
        """Create TensorField object.

        Arguments
        ---------
        x : (N,) array
            Array with x-coordinates.
        y : (M,) array
            Array with y-coordinates.
        g : (M, N, 2, 2)
            Metric grid, using cartesian indexing.
        logx : boolean (optional)
            Convert x -> log10(x), default false.
        logy : boolean (optional)
            Convert y -> log10(y), default false.
        """
        if logx or logy:
            x, y, g = self._log10_converter(x, y, g, logx, logy)
        self.x, self.y, self.g = x, y, g
        self.extent = [x.min(), x.max(), y.min(), y.max()]
        gt = g.transpose((1,0,2,3))  # Cartesian --> matrix indexing
        self.g00 = ip.RectBivariateSpline(x, y, gt[:,:,0,0])
        self.g11 = ip.RectBivariateSpline(x, y, gt[:,:,1,1])
        self.g01 = ip.RectBivariateSpline(x, y, gt[:,:,0,1])
        self.g10 = self.g01

    @staticmethod
    def _log10_converter(x, y, g, logx, logy):
        if logx and x.min()<=0:
            raise ValueError("x-coordinates must be non-negative if logx = True.")
        if logy and y.min()<=0:
            raise ValueError("y-coordinates must be non-negative if logy = True.")
        for i in range(len(y)):
            for j in range(len(x)):
                g[i,j,0,0] *= x[j]**(logx*2)
                g[i,j,1,1] *= y[i]**(logy*2)
                g[i,j,0,1] *= x[j]**logx*y[i]**logy
                g[i,j,1,0] *= x[j]**logx*y[i]**logy
        if logx: x = np.log10(x)
        if logy: y = np.log10(y)
        g /= np.log10(np.e)**2
        return x, y, g

    def __call__(self, x, y, dx = 0, dy = 0):
        g00 = self.g00(x, y, dx = dx, dy = dy)[0,0]
        g11 = self.g11(x, y, dx = dx, dy = dy)[0,0]
        g01 = self.g01(x, y, dx = dx, dy = dy)[0,0]
        g10 = g01
        return np.array([[g00, g01], [g10, g11]])

    def writeto(self, filename):
        np.savez(filename, x=self.x, y=self.y, g=self.g)

    @classmethod
    def fromfile(cls, filename):
        data = np.load(filename)
        return cls(data['x'], data['y'], data['g'])

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
        return np.tensordot(inv_g, Christoffel_1st, (1, 0))

    def _func(self, v, t=0):
        r = np.zeros_like(v)
        G = self.Christoffel_2nd(v[0], v[1])
        r[0] = v[2]
        r[1] = v[3]
        r[2] = -(G[0,0,0]*v[2]*v[2]+G[0,0,1]*v[2]*v[3]+G[0,1,0]*v[3]*v[2]+G[0,1,1]*v[3]*v[3])
        r[3] = -(G[1,0,0]*v[2]*v[2]+G[1,0,1]*v[2]*v[3]+G[1,1,0]*v[3]*v[2]+G[1,1,1]*v[3]*v[3])

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

    def get_contour(self, x0, s0, Npoints = 64, **kwargs):
        """Plot geodesic sphere.

        Arguments
        ---------
        x0 : (2) array
            Central position
        s0 : float
            Geodesic distance
        Npoints : integer (optional)
            Number of points, default 64
        """
        t = np.linspace(0, s0, 30)
        contour = []
        for phi in np.linspace(0, 2*np.pi, Npoints+1):
            g0 = self.__call__(x0[0], x0[1])
            v = np.array([np.cos(phi), np.sin(phi)])
            norm = v.T.dot(g0).dot(v)**0.5
            v /= norm
            s = odeint(self._func, [x0[0], x0[1], v[0], v[1]], t)
            contour.append(s[-1])
            #plt.plot(s[:,0], s[:,1], 'b', lw=0.1)
        return np.array(contour)

    def get_VectorFields(self):
        """Generate vector fields from tensor field.

        Note: The separation and ordering of the two vector fields breaks
        likely down in the presence of singularities.

        Returns
        -------
        vf1, vf2 : VectorFields
        """
        g = self.g
        N, M, _, _ = np.shape(g)
        L1 = np.zeros((N,M))
        L2 = np.zeros((N,M))
        e1 = np.zeros((N,M,2))
        e2 = np.zeros((N,M,2))
        for i in range(N):
            for j in range(M):
                w, v = np.linalg.eig(g[i,j])
                e1[i,j] = v[:,0]
                e2[i,j] = v[:,1]
                L1[i,j] = w[0]
                L2[i,j] = w[1]

        def swap(i,j):
            e_tmp = e1[i,j].copy()
            l_tmp = L1[i,j].copy()
            e1[i,j] = e2[i,j]
            L1[i,j] = L2[i,j]
            e2[i,j] = e_tmp
            L2[i,j] = l_tmp

        # Reorder vector field
        for j in range(0, M):
            if j > 0:
                if abs((e1[0,j]*e1[0,j-1]).sum())<abs((e2[0,j]*e1[0,j-1]).sum()):
                    swap(0,j)
                if (e1[0,j]*e1[0,j-1]).sum() < 0:
                    e1[0,j] *= -1
                if (e2[0,j]*e2[0,j-1]).sum() < 0:
                    e2[0,j] *= -1
            for i in range(1, N):
                if abs((e1[i,j]*e1[i-1,j]).sum())<abs((e2[i,j]*e1[i-1,j]).sum()):
                    swap(i,j)
                if (e1[i,j]*e1[i-1,j]).sum() < 0:
                    e1[i,j] *= -1
                if (e2[i,j]*e2[i-1,j]).sum() < 0:
                    e2[i,j] *= -1

        vf1 = VectorField(self.x, self.y, e1, L2**-0.5)
        vf2 = VectorField(self.x, self.y, e2, L1**-0.5)
        return vf1, vf2

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
        y : 1-D array (M)
            Defines y-grid.
        v : (M, N, 2) array
            Defines vector field on grid, using cartesian indexing.
        d : 2-D array (M,N)
            Defines distance between streamlines.
        """
  
        self.x, self.y, self.v, self.d = x, y, v, d
        self.extent = [x.min(), x.max(), y.min(), y.max()]
        vt = v.transpose((1,0,2))  # Cartesian --> matrix indexing
        dt = d.transpose((1,0))  # Cartesian --> matrix indexing
        self.v0 = ip.RectBivariateSpline(x, y, vt[:,:,0])
        self.v1 = ip.RectBivariateSpline(x, y, vt[:,:,1])
        self.d = ip.RectBivariateSpline(x, y, dt)
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
            # FIXME: what is optimal stepsize?
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
            # FIXME: what is optimal stepsize?
            t = np.linspace(0, -1, 100)
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
    """Test VectorField with mask."""
    plt.figure(figsize=(4,4))

    x = np.linspace(0, 10, 40)
    y = np.linspace(0, 10, 41)
    X, Y = np.meshgrid(x, y)

    v0 = -np.ones_like(X)*1.0
    v1 = np.sin(X/4)*2

    v = np.array([v0, v1])
    v = v.transpose((1,2,0))
    d = np.sqrt(Y)*np.sqrt(X)*0.3+0.02

    vo = np.array([v1, -v0])
    vo = vo.transpose((1,2,0))
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

def test_tf(logscale = False, xinit = [3,3]):
    """Test TensorField and VectorField."""
    plt.figure(figsize=(4,4))

    xinit = np.array(xinit)

    # Generate test-metric
    x = np.linspace(.1, 10, 20)
    y = np.linspace(.1, 10, 20)
    X, Y = np.meshgrid(x, y)
    g00 = np.ones_like(X)
    g10 = np.zeros_like(X)
    g01 = np.zeros_like(X)
    g11 = np.ones_like(X)
    phi = (X-5)*0.3
    beta = 10
    g00 += beta*np.cos(phi)**2
    g01 += beta*np.cos(phi)*np.sin(phi)
    g10 += beta*np.cos(phi)*np.sin(phi)
    g11 += beta*np.sin(phi)**2
    g = np.array([[g00, g01], [g10, g11]])
    g = g.transpose((2,3,0,1))

    tf = TensorField(x, y, g, logx = logscale, logy = logscale)
    tf.writeto('test_tf.npz')
    tf = TensorField.fromfile('test_tf.npz')

    if logscale: xinit = np.log10(xinit)

    contour = tf.get_contour(xinit, 2)
    if logscale: contour = 10**contour
    plt.plot(contour[:,0], contour[:,1], color = 'b', zorder=10)

    vf1, vf2 = tf.get_VectorFields()

    lines = vf1.get_streamlines(xinit, Nmax = 100)
    for line in lines:
        if logscale: line = 10**line
        plt.plot(line.T[0], line.T[1], color='0.5', lw=0.5)

    lines = vf2.get_streamlines(xinit, Nmax = 100)
    for line in lines:
        if logscale: line = 10**line
        plt.plot(line.T[0], line.T[1], color='0.5', lw=0.5)

    plt.xlim([-1,11])
    plt.ylim([-1,11])
    plt.savefig('test_tf.eps')
    quit()

#def test2():
#    g = np.array([[2, 0.0], [0.0, 10]])
#    print eigen(g)
    #plt.quiver(vf1.x, vf1.y, vf1.v[:,:,0], vf1.v[:,:,1])

    #plt.scatter(sample[:,0], sample[:,1], marker='.')
#    for i in range(20):
#        print i
#        sample = tf.relax(sample, spring=False)
    #for i in range(1000):
    #   print i
    #   sample = tf.relax(sample, spring=True)
       #plt.scatter(sample[:,0], sample[:,1], marker='.', zorder=100)

if __name__ == "__main__":
    test_tf()
    test_vf()
