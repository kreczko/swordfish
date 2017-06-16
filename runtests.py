#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import scipy.sparse.linalg as la
import rockfish as RF

def main():
    nside = 32
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, range(npix))
    lon = np.mod(lon+180, 360)-180
    r = (lon**2 + lat**2)**0.5
    bkg = np.exp(-(lat/5)**2.)*100.+1.
    sig = np.exp(-r/10.)*10.+1.
    sig2 = np.exp(-r/10.)
    #sig[30] = 3
    #sig[600] = 3

    flux = [sig, sig2]
    noise = bkg
    sigma = RF.core.Sigma_hpx(nside, sigma=10.)*0.01
    exposure = np.ones(npix)*1e3

    model = RF.Fish(flux, noise, sigma, exposure)
    F, I = RF.core.infoflux(model, solver="cg")
    F = RF.core.effective(F, I, 1)

    #hp.mollview(F)

    q = quantile(F, WeightVec(F), 0.05)
    mask = F > q
    hp.mollview(mask, min = 0, max = 1)

    savefig("test.eps")

    return true

if __name__ == "__main__":
    main()