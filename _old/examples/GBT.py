#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import HARPix as harp
import pylab as plt
from core import *
from tools import *

def CTA_UL():
    # Define energy range
    E = np.logspace(1, 3, 11)  # 10 energy bins
    Ebins = np.array(zip(E[:-1], E[1:]))  # energy bins
    dE = Ebins[:,1]-Ebins[:,0]  # bin width
    Emed = Ebins.prod(axis=1)**0.5  # geometric mean enery
    dims = (len(Emed),)  # HARPix dimensions per pixel

    # Define base pixel size
    nside = 8

    # Signal definition
    spec_sig = 1e-5*np.exp(-(Emed-50)**2/2)*dE  # dN/dE integrated over energy bins (approx.)
    sig = harp.HARPix(dims = dims).add_iso(nside)#.add_singularity((0,0), 1, 20, n = 10)
    sig.add_func(lambda d: spec_sig*np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    # NOTE: sig.data should correpond to the signal intensity (for a fixed DM
    # mass and annihilation cross-section), in units of photons/cm2/s/sr,
    # integrated over the energy bin, but *not* integrated over the pixel size

    # Background definition
    spec_bkg = 1e-3*Emed**-2.7*dE  # dN/dE integrated over energy bins (approx.)
    bkg = harp.zeros_like(sig)
    bkg.add_func(lambda l, b: spec_bkg)
    # NOTE: Same as for the signal.

    # Covariance matrix for energy spectrum uncertainty (ad hoc)
    Sigma = get_sigma(Emed, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))

    # Set up swordfish
    unc = 0.01  # 1% bkg uncertainty
    corr_length = 10  # 10 deg correlation length of bkg uncertainty

    expo = 1e8  # Exposure in cm2 s  (Aeff * Tobs)
    # NOTE: expo can be HARPix object, and should be energy dependent in the
    # end

    fluxes, noise, systematics, exposure = get_model_input([sig], bkg,
            [dict(err = bkg*unc, sigmas = [corr_length], Sigma = Sigma, nside = nside)], expo)
    m = Model(fluxes, noise, systematics, exposure, solver='cg')

    # Calculate upper limits with effective counts method
    ec = EffectiveCounts(m)
    UL = ec.upperlimit(0.05, 0)
    ULg = ec.upperlimit(0.05, 0, gaussian = True)
    s, b = ec.effectivecounts(0, 1.)

    print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
    print "Eff.  signal counts (theta = 1):", s
    print "Eff.  bkg counts (theta = 1)   :", b
    print "Upper limit on theta           :", UL
    print "Upper limit on theta (gaussian):", ULg

    # NOTE: Theta is here defined w.r.t. to the reference signal flux

if __name__ == "__main__":
    CTA_UL()
