#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
import HARPix as harp
import pylab as plt

class Model(object):
    def __init__(self, flux, noise, systematics, exposure, solver = 'direct'):
        self.flux = flux
        self.noise = noise
        self.exposure = exposure

        self.solver = solver
        self.nbins = len(self.noise)  # Number of bins
        self.ncomp = len(self.flux)   # Number of flux components
        if systematics is not None:
            self.systematics = la.aslinearoperator(systematics)
        else:
            self.systematics = la.LinearOperator(
                    (self.nbins, self.nbins), matvec = lambda x: x*0.)

    def solveD(self, thetas = None, psi = 1.):
        # FIXME: Cache results
        noise = self.noise
        exposure = self.exposure*psi
        if thetas is not None: 
            for i in range(max(self.ncomp, len(thetas))):
                noise += thetas[i]*self.flux[i]
        D = (
                la.aslinearoperator(sp.diags(self.noise/self.exposure))
                + self.systematics
                )
        x = np.zeros((self.ncomp, self.nbins))
        if self.solver == "direct":
            #print 'direct'
            dense = D(np.eye(self.nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self.ncomp):
                x[i] = np.dot(invD, self.flux[i])
        elif self.solver == "cg":
            #print 'cg'
            def callback(x):
                pass
                #print sum(x), len(x)
            for i in range(self.ncomp):
                x[i] = la.cg(D, self.flux[i], callback = callback, tol = 1e-5)[0]
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

    def effectivecounts(self, i, theta, psi = 1.):
        I0 = self.model.effectivefishermatrix(i, psi = psi)
        thetas = np.zeros(self.model.ncomp)
        thetas[i] = theta
        I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
        s = 1/(1/I-1/I0)*theta**2
        b = 1/I0/(1/I-1/I0)**2*theta**2
        return s, b

    def upperlimit(self, alpha, i, psi = 1., gaussian = True):
        Z = 2.64  # FIXME
        I0 = self.model.effectivefishermatrix(i, psi = psi)
        if gaussian:
            return Z/np.sqrt(I0)
        else:
            thetas = np.zeros(self.model.ncomp)
            thetaUL_est = Z/np.sqrt(I0)  # Gaussian estimate
            thetas[i] = thetaUL_est
            I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
            if (I0-I)<0.01*I:
                thetaUL = thetaUL_est
            else:
                raise NotImplementedError()  # FIXME Finish implementation
                theta_list = np.linspace(thetaUL_est/1000, thetaUL_est*1000, 1000)
                z_list = []
                for theta in theta_list:
                    s, b = self.effectivecounts(i, theta = theta, psi = psi)
                    z_list.append(s/np.sqrt(s+b))
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

def Sigma_hpx(nside, sigma=0., scale=1.):
    npix = hp.nside2npix(nside)
    def hpxconvolve(x):
        if sigma != 0.:
            X = scale*x
            if (scale*x).ndim != 1:
                X = (scale*x).reshape((scale*x).shape[1],npix)
            alm = hp.map2alm(X)
            x = hp.alm2map(alm, nside, sigma = np.deg2rad(sigma), verbose=False)
            return x*scale
    def flat(x):
        return scale*sum(x*scale)
    if sigma == np.Inf:
        return la.LinearOperator((npix, npix), matvec = lambda x: flat(x))
    else:
        return la.LinearOperator((npix, npix), matvec = lambda x: hpxconvolve(x))

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

    def add_systematics(self, variance = None, sigmas = None, Sigma = None,
            nside = None):
        variance_data = variance.get_formatted_like(self.harpix).data
        F = np.sqrt(variance_data)
        self.Flist.append(F)

        lmax = 3*nside - 1  # default from hp.smoothing
        Nalm = hp.Alm.getsize(lmax)
        G = np.zeros(Nalm, dtype = 'complex128')
        for sigma in sigmas:
            H = hp.smoothalm(np.ones(Nalm), sigma = np.deg2rad(sigma), inplace = False, verbose = False)
            npix = hp.nside2npix(nside)
            m = np.zeros(npix)
            m[0] = 1
            M = hp.smoothing(m, sigma = np.deg2rad(sigma))
            G += H/max(M)
        G /= len(sigmas)
        self.Glist.append(G)
        T1 = harp.get_trans_matrix(self.harpix, nside, nested = False)
        T2 = harp.get_trans_matrix(nside, self.harpix, nested = False)
        self.Tlist.append([T1, T2])
        self.nsidelist.append(nside)

        self.Slist.append(Sigma)

    def _matvec(self,x):
        result = np.zeros(self.N)
        for nside, F, G, S, Ts in zip(self.nsidelist, self.Flist, self.Glist, self.Slist, self.Tlist):
            y = x.reshape((-1,)+self.harpix.dims)*F
            z = harp.trans_data(Ts[0], y)
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
                alm = hp.map2alm(a)
                alm *= G
                b = hp.alm2map(alm, nside, verbose = False)
            c = harp.trans_data(Ts[1], b)

            #hp.mollview(c, nest=True)
            #plt.savefig('test.eps')
            #quit()

            d = c.reshape((-1,)+self.harpix.dims)*F
            result += d.flatten()
        return result

#    def _flat(self, x):
#        return self.scale*sum(x*self.scale)

#    def _hpxconvolve(self, x):
#        if self.sigma != 0.:
#            X = self.scale*x
#            if (self.scale*x).ndim != 1:
#                X = (self.scale*x).reshape((self.scale*x).shape[1],self.npix)
#            alm = hp.map2alm(X)
#            x = hp.alm2map(alm, self.nside, sigma = np.deg2rad(self.sigma), verbose=False)
#            return x*self.scale

def Sigma_hpx(nside, sigma=0., scale=1.):
    npix = hp.nside2npix(nside)
    def hpxconvolve(x):
        if sigma != 0.:
            X = scale*x
            if (scale*x).ndim != 1:
                X = (scale*x).reshape((scale*x).shape[1],npix)
            alm = hp.map2alm(X)
            x = hp.alm2map(alm, nside, sigma = np.deg2rad(sigma), verbose=False)
            return x*scale
    def flat(x):
        return scale*sum(x*scale)
    if sigma == np.Inf:
        return la.LinearOperator((npix, npix), matvec = lambda x: flat(x))
    else:
        return la.LinearOperator((npix, npix), matvec = lambda x: hpxconvolve(x))

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
    sig.data += 1.0  # EGBG
    plot_harp(sig, 'sig.eps')
    sig.mul_sr()
    #sig.print_info()

    # Background definition
    bg = harp.zeros_like(sig)
    bg.add_func(lambda l, b: 0./(b**2+1.)**0.5+0.1)
    plot_harp(bg, 'bg.eps')
    bg.mul_sr()
    #bg.print_info()

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    var = bg*bg
    var.data *= 0.01  # 10% uncertainty
    cov.add_systematics(variance = var, sigmas = [20.,], Sigma = None, nside = 16)

    data = np.ones(len(bg.data))
    print cov.dot(data)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*1000
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

#    I = m.fishermatrix()
#    F = m.infoflux()
#    I = m.effectivefishermatrix(0)
    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    f.div_sr()
    plot_harp(f, 'test.eps')
    quit()

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

#    I = m.fishermatrix()
#    F = m.infoflux()
#    I = m.effectivefishermatrix(0)
    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    plot_harp(f, 'test.eps')
    quit()

#    ec = EffectiveCounts(m)
#    print ec.effectivecounts(0, 3.00)
#    print ec.upperlimit(1, 0)

def test_spectra():
    import pylab as plt

    x = np.linspace(0, 10, 1000)  # energy
    dx = x[1]-x[0]  # bin size

    fluxes = [(np.exp(-(x-5)**2/2/3**2)+np.exp(-(x-5)**2/2/0.2**2))*dx]
    noise = (1+x*0.0001)*dx
    exposure = np.ones_like(noise)*10000.0
    X, Y = np.meshgrid(x,x)
    systematics = 0.11*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/40**2) + np.exp(-(X-Y)**2/2/20**2)
                #np.exp(-(X-Y)**2/2/10**2) + np.exp(-(X-Y)**2/2/5**2) +
                #np.exp(-(X-Y)**2/2/2**2) + np.exp(-(X-Y)**2/2/1**2)
                )).dot(np.diag(noise))
    print systematics.dot(np.ones_like(x))
    quit()
    m = Model(fluxes, noise, systematics, exposure, solver='cg')
    f = m.effectiveinfoflux(0)
    plt.plot(np.sqrt(fluxes[0]**2/dx/noise))
    plt.plot(np.sqrt(f/dx), label='Info flux')
    plt.legend()
    plt.savefig('test.eps')

if __name__ == "__main__":
    test_simple()
    #test_spectra()
