#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la

class Fish(object):
    def __init__(self, flux, noise, systematics, exposure):
        self.flux = flux
        self.noise = noise
        self.systematics = systematics
        self.exposure = exposure

    def infomatrix(self, solver = "direct"):
        D = opDiagonal(self.noise/self.exposure) + model.systematics
        n = length(model.flux)
        x = np.zeros(n)
        if solver == "direct":
            invD = np.inv(np.full(D))
            for i in range(n):
                x[i] = np.invD*self.flux[i]
        elif solver == "cg":
            for i in range(n):
                Pl = np.diagonal(np.diagonal(D))
                x[i] = la.cg(D, self.flux[i], Pl = Pl, verbose=True, maxiter=10)
        else:
            error("Solver unknown.")
        I = np.zeros(n,n)
        for i in range(n):
            for j in range(i):
                tmp = sum(model.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I

    def infoflux(self, solver = "direct", maxiter = 10, **kwargs):
        # D = np.zeros((len(self.noise), len(self.noise)))
        # self.systematics is a linear operator therefore 
        # TypeError: unsupported operand type(s) for +: 'NoneType' and '_ScaledLinearOperator'
        # FIXME: Not checked if this is actually performing operations as we want
        def D(x):
            D_temp = np.zeros((len(self.noise), len(self.noise)))
            np.fill_diagonal(D_temp, self.noise/self.exposure)
            return D_temp.dot(np.ones(len(x)))
        D_1 = la.LinearOperator((len(self.noise), len(self.noise)), matvec=D)
        D = D_1 + self.systematics
        #######################################
        n = len(self.flux)
        # x = np.zeros((n,len(self.flux[0])))
        x = np.empty(shape=(n,), dtype=object)    
        if solver == "direct":
            invD = np.inv(full(D))
            for i in range(n):
                x[i] = invD*self.flux[i]
        elif solver == "cg":
            # FIXME: Not quite sure what this is doing right now
            for i in range(n):
                # Note: la.cg has tuple output so we just take result i.e. [0]
                x[i] = la.cg(D, self.flux[i], maxiter = maxiter)[0]
        else:
            raise KeyError("Solver unknown.")
        I = np.empty(shape=(n,n), dtype=object)    
        F = np.empty(shape=(n,n), dtype=object)    
        # I = np.zeros((n,n,len(self.flux[0])))
        # F = np.zeros((n,n,len(self.flux[0])))
        for i in range(n):
            for j in range(i):
                tmp = x[i]*x[j]*self.noise/(self.exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
                tmp = sum(self.flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return F, I

def effective(I, i):
    invI = np.inv(I)
    return 1/invI[i,i]

def effective(F, I, i):
    n = np.size(I, 1)
    if n == 1:
        return F[i,i]
    indices = np.setdiff1d(range(n), i)
    # Code now runs till here :)
    eff_F = F[i,i]
    C = np.zeros(n-1)
    B = np.zeros((n-1,n-1))
    for j in range(n):
        C[j] = I[indices[j],i]
        for k in range(n-1):
            B[j,k] = I[indices[j], indices[k]]
    invB = inv(B)
    for j in range(n):
        for k in range(n):
            eff_F = eff_F - F[i,indices[j]]*invB[j,k]*C[k]
            eff_F = eff_F - C[j]*invB[j,k]*F[indices[k],i]
            for l in range(n):
                for m in range(n):
                    eff_F = eff_F + C[j]*invB[j,l]*F[indices[l],indices[m]]*invB[m,k]*C[k]
    return eff_F

def tensorproduct(Sigma1, Sigma2):
    return tensorproduct(la.LinearOperator(Sigma1), Sigma2)

def tensorproduct(Sigma1, Sigma2):
    return tensorproduct(Sigma1, full(Sigma2))

def tensorproduct(Sigma1, Sigma2):
    n1 = size(Sigma1,1)
    n2 = size(Sigma2,1)
    N = n1*n2
    def Sigma(x):
        A = np.reshape(x, (n1, n2))
        B = np.zeros(A)
        for i in range(n2):
            y = Sigma1*A[:,i]
            for j in range(n2):
                B[:,j] += Sigma2[i,j]*y 
        return np.reshape(B, N)
    return la.LinearOperator(N, N, True, True, lambda x: Sigma(x))

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
