#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
import HARPix as harp
import pylab as plt
from copy import deepcopy

class Model(object):  # Everything is flux!
    def __init__(self, flux, noise, systematics, exposure, solver = 'direct'):
        self.flux = flux
        self.noise = noise
        self.exposure = exposure
        self.cache = None

        self.solver = solver
        self.nbins = len(self.noise)  # Number of bins
        self.ncomp = len(self.flux)   # Number of flux components
        if systematics is not None:
            self.systematics = la.aslinearoperator(systematics)
        else:
            self.systematics = la.LinearOperator(
                    (self.nbins, self.nbins), matvec = lambda x: x*0.)

    def solveD(self, thetas = None, psi = 1.):
        noise = deepcopy(self.noise)
        exposure = self.exposure*psi
        if thetas is not None: 
            for i in range(max(self.ncomp, len(thetas))):
                noise += thetas[i]*self.flux[i]
        D = (
                la.aslinearoperator(sp.diags(noise/exposure))
                + self.systematics
                )
        x = np.zeros((self.ncomp, self.nbins))
        if self.solver == "direct":
            dense = D(np.eye(self.nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self.ncomp):
                x[i] = np.dot(invD, self.flux[i])
        elif self.solver == "cg":
            def callback(x):
                pass
                #print len(x), sum(x), np.mean(x)
            for i in range(self.ncomp):
                x[i] = la.cg(D, self.flux[i], x0 = self.cache, callback = callback, tol = 1e-5)[0]
                self.cache= x[i]
        else:
            raise KeyError("Solver unknown.")
        return x, noise, exposure

    def fishermatrix(self, thetas = None, psi = 1.):
        x, noise, exposure = self.solveD(thetas=thetas, psi=psi)
        I = np.zeros((self.ncomp,self.ncomp))
        for i in range(self.ncomp):
            for j in range(i+1):
                tmp = sum(self.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I

    def infoflux(self, thetas = None, psi = 1.):
        x, noise, exposure = self.solveD(thetas=thetas, psi=psi)

        F = np.zeros((self.ncomp,self.ncomp,self.nbins))
        for i in range(self.ncomp):
            for j in range(i+1):
                tmp = x[i]*x[j]*noise/(exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
        return F

    def effectivefishermatrix(self, i, **kwargs):
        I = self.fishermatrix(**kwargs)
        invI = np.linalg.linalg.inv(I)
        return 1./invI[i,i]

    def effectiveinfoflux(self, i, **kwargs):
        F = self.infoflux(**kwargs)
        I = self.fishermatrix(**kwargs)
        n = self.ncomp
        if n == 1:
            return F[i,i]
        indices = np.setdiff1d(range(n), i)
        eff_F = F[i,i]
        C = np.zeros(n-1)
        B = np.zeros((n-1,n-1))
        for j in range(n-1):
            C[j] = I[indices[j],i]
            for k in range(n-1):
                B[j,k] = I[indices[j], indices[k]]
        invB = np.linalg.linalg.inv(B)
        for j in range(n-1):
            for k in range(n-1):
                eff_F = eff_F - F[i,indices[j]]*invB[j,k]*C[k]
                eff_F = eff_F - C[j]*invB[j,k]*F[indices[k],i]
                for l in range(n-1):
                    for m in range(n-1):
                        eff_F = eff_F + C[j]*invB[j,l]*F[indices[l],indices[m]]*invB[m,k]*C[k]
        return eff_F

class EffectiveCounts(object):
    def __init__(self, model):
        self.model = model

    def counts(self, i, theta):
        return sum(self.model.flux[i]*self.model.exposure*theta)

    def effectivecounts(self, i, theta, psi = 1.):
        I0 = self.model.effectivefishermatrix(i, psi = psi)
        thetas = np.zeros(self.model.ncomp)
        thetas[i] = theta
        I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
        if I0 == I:
            return 0., None
        s = 1/(1/I-1/I0)*theta**2
        b = 1/I0/(1/I-1/I0)**2*theta**2
        return s, b

    def upperlimit(self, alpha, i, psi = 1., gaussian = False):
        Z = 2.64  # FIXME
        I0 = self.model.effectivefishermatrix(i, psi = psi)
        if gaussian:
            return Z/np.sqrt(I0)
        else:
            thetas = np.zeros(self.model.ncomp)
            thetaUL_est = Z/np.sqrt(I0)  # Gaussian estimate
            thetas[i] = thetaUL_est
            I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
            if (I0-I)<0.02*I:  # 1% accuracy of limits
                thetaUL = thetaUL_est
            else:
                z_list = []
                theta_list = [thetaUL_est/1]
                while True:
                    theta = theta_list[-1]
                    s, b = self.effectivecounts(i, theta = theta, psi = psi)
                    z_list.append(s/np.sqrt(s+b))
                    if z_list[-1] > Z:
                        break
                    else:
                        theta_list.append(theta*1.5)
                thetaUL = np.interp(Z, z_list, theta_list)
            return thetaUL

    def discoveryreach(self, alpha, i, psi = 1., gaussian = False):
        raise NotImplemented()

class Visualization(object):
    def __init__(self, xy, I11, I22, I12):
        pass

    def plot(self):
        pass

    def integrate(self):
        pass

def tensorproduct(Sigma1, Sigma2):
    Sigma1 = la.aslinearoperator(Sigma1)
    Sigma2 = la.aslinearoperator(Sigma2)
    n1 = np.shape(Sigma1)[0]
    n2 = np.shape(Sigma2)[0]
    Sigma2 = Sigma2(np.eye(n2))
    N = n1*n2
    def Sigma(x):
        A = np.reshape(x, (n1, n2))
        B = np.zeros_like(A)
        for i in range(n2):
            y = Sigma1(A[:,i])
            for j in range(n2):
                B[:,j] += Sigma2[i,j]*y
        return np.reshape(B, N)
    return la.LinearOperator((N, N), matvec = lambda x: Sigma(x))

class HARPix_Sigma(la.LinearOperator):
    """docstring for CovarianceMatrix"""
    def __init__(self, harpix):
        self.harpix = harpix
        self.N = np.prod(np.shape(self.harpix.data))
        super(HARPix_Sigma, self).__init__(None, (self.N, self.N))
        self.Flist = []
        self.Glist = []
        self.Slist = []
        self.Tlist = []
        self.nsidelist = []

    def add_systematics(self, err = None, sigmas = None, Sigma = None,
            nside = None):
        F = err.get_formatted_like(self.harpix).get_data(mul_sr=True)
        self.Flist.append(F)

        lmax = 3*nside - 1  # default from hp.smoothing
        Nalm = hp.Alm.getsize(lmax)
        G = np.zeros(Nalm, dtype = 'complex128')
        for sigma in sigmas:
            H = hp.smoothalm(np.ones(Nalm), sigma = np.deg2rad(sigma), inplace = False, verbose = False)
            npix = hp.nside2npix(nside)
            m = np.zeros(npix)
            m[10] = 1
            M = hp.alm2map(hp.map2alm(m)*H, nside, verbose = False)
            G += H/max(M)
        G /= len(sigmas)
        self.Glist.append(G)
        T = harp.get_trans_matrix(self.harpix, nside, nest = False, counts = True)
        self.Tlist.append(T)
        self.nsidelist.append(nside)

        self.Slist.append(Sigma)

    def _matvec(self,x):
        result = np.zeros(self.N)
        for nside, F, G, S, T in zip(self.nsidelist, self.Flist, self.Glist, self.Slist, self.Tlist):
            y = x.reshape((-1,)+self.harpix.dims)*F
            z = harp.trans_data(T, y)
            if S is not None:
                a = S.dot(z.T).T
            else:
                a = z
            b = np.zeros_like(z)
            if self.harpix.dims is not ():
                for i in range(self.harpix.dims[0]):
                    alm = hp.map2alm(a[:,i])
                    alm *= G
                    b[:,i] = hp.alm2map(alm, nside, verbose = False)
            else:
                alm = hp.map2alm(a, iter = 0)  # Older but faster routine
                alm *= G
                b = hp.alm2map(alm, nside, verbose = False)
            c = harp.trans_data(T.T, b)
            d = c.reshape((-1,)+self.harpix.dims)*F
            result += d.flatten()
        return result

def get_sigma(x, f):
    X, Y = np.meshgrid(x,x)
    Sigma = f(X,Y)
    A = 1/np.sqrt(np.diag(Sigma))
    Sigma = np.diag(A).dot(Sigma).dot(np.diag(A))
    return Sigma

def test_3d():
    from HARPix import HARPix
    import healpy as hp
    import pylab as plt

    def plot_harp(h, filename, dims = ()):
        m = h.get_healpix(128, idxs= dims)
        hp.mollview(m, nest=True)
        plt.savefig(filename)

    def plot_rock(h, filename):
        nside = 32
        N = h.dims[0]
        npix = hp.nside2npix(nside)
        rings = hp._pixelfunc.pix2ring(nside, np.arange(npix), nest = True)
        T = harp.get_trans_matrix(h, nside)
        data = harp.trans_data(T, h.data)
        out = []
        for r in range(1, 4*nside):
            mask = rings == r
            out.append(data[mask].sum(axis=0))
        out = np.array(out)
        plt.imshow(out, aspect=0.8*N/(4*nside))
        plt.xlabel("Energy [AU]")
        plt.ylabel("Latitude [AU]")
        plt.savefig(filename)

    x = np.linspace(0, 10, 20)
    nside = 16

    dims = (len(x),)

    # Signal definition
    spec_sig = np.exp(-(x-5)**2/2)
    sig = HARPix(dims = dims).add_iso(nside).add_singularity( (50,50), 1, 20, n = 10)
    sig.add_func(lambda d: spec_sig*np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.add_func(lambda d: spec_sig/(d+1)**1, mode = 'dist', center=(50,50))
    sig.mul_sr()

    # Background definition
    bg = harp.zeros_like(sig)
    spec_bg = x*0. + 1.
    bg.add_func(lambda l, b: spec_bg*(0./(b**2+1.)**0.5+0.1))
    bg.mul_sr()

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    corr = lambda x, y: np.exp(-(x-y)**2/2/3**2)
    Sigma = get_sigma(x, corr)
    cov.add_systematics(err = bg*0.1, sigmas = [20.,], Sigma = Sigma, nside = 16)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*100.0
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    f.div_sr()
    plot_harp(f, 'test.eps', dims = (11,))
    plot_rock(f, 'test.eps')

def get_model_input(signals, noise, systematics, exposure):
    # Everything is intensity
    S = [sig.get_formatted_like(signals[0]).get_data(mul_sr=True).flatten() for sig in signals]
    N = noise.get_formatted_like(signals[0]).get_data(mul_sr=True).flatten()
    SYS = HARPix_Sigma(signals[0])
    if systematics is None:
        SYS = None
    else:
        for sys in systematics:
            SYS.add_systematics(**sys)
    if isinstance(exposure, float):
        E = np.ones_like(N)*exposure
    else:
        E = exposure.get_formatted_like(signals[0]).get_data().flatten()
    return S, N, SYS, E

def test_UL():
    from HARPix import HARPix
    import healpy as hp
    import pylab as plt

    # Signal definition
    sig = HARPix().add_iso(16)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))

    # Background definition
    bg = harp.zeros_like(sig)
    bg.add_func(lambda l, b: 0./(b**2+1.)**0.5+1.0)

    fluxes, noise, systematics, exposure = get_model_input(
            [sig], bg, [dict(err=bg*0.1, sigmas = [20.,], Sigma = None, nside =
                16)], bg*100.)
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

    I = m.fishermatrix()
    print I

    ec = EffectiveCounts(m)
    UL = ec.upperlimit(0.05, 0)
    ULg = ec.upperlimit(0.05, 0, gaussian = True)
    s, b = ec.effectivecounts(0, 1.0)

    print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
    print "Eff.  signal counts (theta = 1):", s
    print "Eff.  bkg counts (theta = 1)   :", b
    print "Upper limit on theta           :", UL
    print "Upper limit on theta (gaussian):", ULg

def test_simple():
    from HARPix import HARPix
    import healpy as hp
    import pylab as plt

    def plot_harp(h, filename):
        m = h.get_healpix(128)
        hp.mollview(m, nest=True)
        plt.savefig(filename)

    dims = ()
    nside = 16

    # Signal definition
    spec = 1.
    sig = HARPix(dims = dims).add_iso(nside).add_singularity( (50,50), .1, 20, n = 100)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.add_func(lambda d: 1/(d+1)**1, mode = 'dist', center=(50,50))
    sig.data += 0.1  # EGBG
    plot_harp(sig, 'sig.eps')
    sig.mul_sr()

    # Background definition
    bg = harp.zeros_like(sig)
    bg.add_func(lambda l, b: 0./(b**2+1.)**0.5+1.0)
    plot_harp(bg, 'bg.eps')
    bg.mul_sr()
    #bg.print_info()

    # Covariance matrix definition

    cov = HARPix_Sigma(sig)
    cov.add_systematics(err = bg*0.1, sigmas = [20.,], Sigma = None, nside = 64)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*10000.
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0, thetas = [0.000], psi = 1.)
    f = harp.HARPix.from_data(sig, F)
    f.div_sr()
    plot_harp(f, 'test.eps')
    # quit()

def test_MW_dSph():
    from HARPix import HARPix
    import healpy as hp
    import pylab as plt

    def plot_harp(h, filename):
        m = h.get_healpix(128)
        hp.mollview(np.log10(m), nest=True)
        plt.savefig(filename)

    dims = ()

    # Signal definition
    spec = 1.
    MW = HARPix(dims = dims).add_iso(8).add_singularity((0,0), 0.1, 20, n = 100)
    MW.add_func(lambda d: spec/(.1+d)**2, mode = 'dist', center=(0,0))
    pos = (50, 40)
    dSph = HARPix(dims = dims).add_singularity(pos, 0.1, 20, n = 100)
    dSph.add_func(lambda d: 0.1*spec/(.1+d)**2, mode = 'dist', center=pos)
    sig = MW + dSph
    sig.data += 1  # EGBG
    plot_harp(sig, 'sig.eps')

    # Background definition
    bg = HARPix(dims = dims).add_iso(64)
    bg.add_func(lambda l, b: 1/(b+1)**2)
    plot_harp(bg, 'bg.eps')

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    var = bg*bg
    var.data *= 0.1  # 10% uncertainty
    cov.add_systematics(variance = var, sigmas = [100], Sigma = None, nside =
            nside)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.get_formatted_like(sig).data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*1.
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    plot_harp(f, 'test.eps')

def test_spectra():
    import pylab as plt

    x = np.linspace(0, 10, 1000)  # energy
    dx = x[1]-x[0]  # bin size

    fluxes = [(np.exp(-(x-5)**2/2/3**2)+np.exp(-(x-5)**2/2/0.2**2))*dx]
    noise = (1+x*0.0001)*dx
    exposure = np.ones_like(noise)*1.
    X, Y = np.meshgrid(x,x)
    systematics = 0.1*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/40**2) + np.exp(-(X-Y)**2/2/20**2)
                )).dot(np.diag(noise))
    m = Model(fluxes, noise, systematics, exposure, solver='cg')
    f = m.effectiveinfoflux(0)
    plt.plot(np.sqrt(fluxes[0]**2/dx/noise))
    plt.plot(np.sqrt(f/dx), label='Info flux')
    plt.legend()
    plt.savefig('test.eps')

def smoothtest():
    nside = 64
    sigma = 30

    lmax = 3*nside - 1  # default from hp.smoothing
    lmax += 10
    Nalm = hp.Alm.getsize(lmax)
    H = hp.smoothalm(np.ones(Nalm), sigma = np.deg2rad(sigma), inplace = False)
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    m[1000] = 1
    M = hp.alm2map(hp.map2alm(m, lmax = lmax)*H, nside, lmax = lmax)
    G = H/max(M)
    m *= 0
    m[0] = 1
    I = hp.alm2map(hp.map2alm(m, lmax = lmax)*G, nside, lmax = lmax)
    print max(I)
    hp.mollview(I)
    plt.savefig('test.eps')

def test_covariance():
    from HARPix import HARPix
    import healpy as hp
    import pylab as plt

    nside = 8

    # Signal definition
    sig = HARPix().add_iso(nside).add_singularity( (50,50), .1, 50, n = 100)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.mul_sr()

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    bg = sig*.0
    bg.data += 1
    cov.add_systematics(err = bg*0.1,
            sigmas = [20.,], Sigma = None, nside = nside)

    x = np.zeros_like(sig.data)
    x[901] = 1.
    y = cov.dot(x)
    sig.data = y
    sig.div_sr()
    z = sig.get_healpix(128)
    hp.mollview(z, nest = True)
    plt.savefig('test.eps')

def test_matrix():
    import pylab as plt

    x = np.linspace(0, 10, 1000)  # energy
    dx = x[1]-x[0]  # bin size

    fluxes = [np.exp(-(x-5)**2/2/3**2)*dx,np.exp(-(x-5)**2/2/0.2**2)*dx]
    noise = np.ones_like(x)*dx
    exposure = np.ones_like(noise)*1.001
    X, Y = np.meshgrid(x,x)
    systematics = 0.1*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/40**2) + np.exp(-(X-Y)**2/2/20**2)
                )).dot(np.diag(noise))
    m = Model(fluxes, noise, systematics, exposure, solver='cg')
    I = m.fishermatrix()
    print I
    F = m.infoflux()
    f = m.effectiveinfoflux(0)
    plt.plot(x, f*dx, label='Feff')
    plt.plot(x, F[0,0]*dx, label='F00')
    plt.plot(x, F[1,1]*dx, label='F11')
    plt.plot(x, F[0,1]*dx, label='F01')
    plt.legend()
    plt.savefig('test.eps')


if __name__ == "__main__":
    #test_3d()
    #test_covariance()
    #test_simple()
    test_UL()
    #smoothtest()
    #test_spectra()
    # test_matrix()
