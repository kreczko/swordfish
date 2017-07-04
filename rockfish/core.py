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
        self.exposure = exposure

        self.solver = "direct"
        self.nbins = len(self.noise)  # Number of bins
        self.ncomp = len(self.flux)   # Number of flux components
        if systematics is not None:
            self.systematics = la.aslinearoperator(systematics)
        else:
            self.systematics = la.LinearOperator(
                    (self.nbins, self.nbins), matvec = lambda x: x*0.)

    def solveD(self, theta = None, psi = 1.):
        # FIXME: Cache results
        noise = self.noise
        exposure = self.exposure*psi
        if theta is not None: 
            for i in range(max(self.ncomp, len(theta))):
                noise += theta[i]*self.flux[i]
        D = (
                la.aslinearoperator(sp.diags(self.noise/self.exposure))
                + self.systematics
                )
        x = np.zeros((self.ncomp, self.nbins))
        if self.solver == "direct":
            dense = D(np.eye(self.nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self.ncomp):
                x[i] = np.dot(invD, self.flux[i])
        elif self.solver == "cg":
            for i in range(self.ncomp):
                x[i] = la.cg(D, self.flux[i], **kwargs)[0]
        else:
            raise KeyError("Solver unknown.")
        return x, noise, exposure

    def fishermatrix(self, theta = None, psi = 1.):
        x, noise, exposure = self.solveD(theta=theta, psi=psi)
        I = np.zeros((self.ncomp,self.ncomp))
        for i in range(self.ncomp):
            for j in range(i+1):
                tmp = sum(self.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I

    def infoflux(self, theta = None, psi = 1.):
        x, noise, exposure = self.solveD(theta=theta, psi=psi)

        F = np.zeros((self.ncomp,self.ncomp,self.nbins))
        for i in range(self.ncomp):
            for j in range(i+1):
                tmp = x[i]*x[j]*noise/(exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
        return F

    def effectivefishermatrix(self, i, **kwargs):
        I = self.infomatrix(**kwargs)
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
        thetas = zeros(I0.ncomp)
        thetas[i] = theta
        I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
        s = 1/(1/I-1/I0)
        b = 1/I/(1/I-1/I0)**2
        return s, b

    def upperlimit(self, alpha, i, psi = 1., gaussian = False):
        Z = Z(alpha)
        I0 = self.effectivefishermatrix(i, psi = psi)
        if gaussian:
            return Z/sqrt(I0)
        else:
            thetas = zeros(I0.ncomp)
            thetas[i] = theta
            I = self.model.effectivefishermatrix(i, thetas = thetas, psi = psi)
            if (I-I0)<0.01*I:
                return Z/sqrt(I0)
            else:
                raise NotImplemented()

    def discoveryreach(self, alpha, i, psi = 1., gaussian = False):
        raise NotImplemented()

#class NestedSignalModel(object):
#    def __init__(self, model, pmodels, pshape):
#        self.model = model
#        self.pmodels = pmodels
#        self.pshape= pshape
#
#    def effectivefishermatrix(self, **kwargs):
#        I = self.model.effectivefishermatrix(**kwargs)
#        for j, pmodel in self.pmodels:
#            pI = pmodel.effectivefishermatrix(**kwargs)
#            I += pI
#        return I
#
#    def effectiveinfoflux(self, **kwargs):
#        F = self.model.effectiveinfoflux(**kwargs)
#        F = F.reshape((-1,)+self.pshape)
#        for j, pmodel in self.pmodels:
#            pF = pmodel.effectiveinfoflux(**kwargs)
#            pF = pF.reshape((self.pshape)+(-1,))
#            pF = pF.sum(axis=-1)
#            F[j] += pF
#        return F.flatten()

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
