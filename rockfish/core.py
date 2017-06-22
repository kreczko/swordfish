#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp

class Model(object):
    def __init__(self, flux, noise, systematics, exposure):
        self.flux = flux
        self.noise = noise
        self.systematics = la.aslinearoperator(systematics)
        self.exposure = exposure

    def infomatrix(self, solver = "direct", **kwargs):
        n = len(self.flux)   # Number of flux components
        N = len(self.noise)  # Number of bins
        D = (
                la.aslinearoperator(sp.diags(self.noise/self.exposure))
                + self.systematics
                )
        x = np.zeros((n, N))
        if solver == "direct":
            dense = D(np.eye(N))
            invD = np.linalg.linalg.inv(dense)
            for i in range(n):
                x[i] = np.dot(invD, self.flux[i])
        elif solver == "cg":
            for i in range(n):
                x[i] = la.cg(D, self.flux[i], **kwargs)[0]
        else:
            raise KeyError("Solver unknown.")
        I = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1):
                tmp = sum(self.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I

    def infoflux(self, solver = "direct", **kwargs):
        n = len(self.flux)   # Number of flux components
        N = len(self.noise)  # Number of bins
        D = (
                la.aslinearoperator(sp.diags(self.noise/self.exposure))
                + self.systematics
                )
        x = np.zeros((n, N))
        if solver == "direct":
            dense = D(np.eye(N))
            invD = np.linalg.linalg.inv(dense)
            for i in range(n):
                x[i] = np.dot(invD, self.flux[i])
        elif solver == "cg":
            for i in range(n):
                x[i] = la.cg(D, self.flux[i], **kwargs)[0]
        else:
            raise KeyError("Solver unknown.")
        I = np.zeros((n,n))
        F = np.zeros((n,n,N))
        for i in range(n):
            for j in range(i+1):
                tmp = x[i]*x[j]*self.noise/(self.exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
                tmp = sum(self.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return F, I

    def effectiveinfomatrix(self, i, **kwargs):
        I = self.infomatrix(**kwargs)
        invI = np.linalg.linalg.inv(I)
        return 1./invI[i,i]

    def effectiveinfoflux(self, i, **kwargs):
        F, I = self.infoflux(**kwargs)
        n = np.shape(I)[0]
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
    def __init__(self, rfmodel):
        self.rfmodel = rfmodel

    def effectivecounts(self, i, theta):
        I = self.rfmodel.effectiveinfomatrix(i)
        noise = self.rfmodel.noise - self.flux[i]
        I0 = self.rfmodel.effectiveinfomatrix(i, noise = noise)
        s = 1/(1/I-1/I0)
        b = 1/I/(1/I-1/I0)**2
        return s, b

    def upperlimit(self, alpha, i, gaussian = False):
        Z = Z(alpha)
        I = self.effectiveinfomatrix(i)
        if gaussian:
            return Z/sqrt(I)
        else:
            I0 = self.effectiveinfomatrix(i, noise = self.noise - self.flux[i])
            if (I-I0)<0.01*I:
                return Z/sqrt(I)
            else:
                raise NotImplemented()

    def discoveryreach(self, alpha, i, gaussian = False):
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
            alm = hp.map2alm(x*scale)
            x = hp.alm2map(alm, nside, sigma = np.deg2rad(sigma), verbose=False)
            return x*scale
    def flat(x):
        return scale*sum(x*scale)
    if sigma == np.Inf:
        return la.LinearOperator((npix, npix), matvec = lambda x: flat(x))
    else:
        return la.LinearOperator((npix, npix), matvec = lambda x: hpxconvolve(x))
