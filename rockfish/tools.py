#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import HARPix as harp
from scipy.sparse import linalg as la
from core import *

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



