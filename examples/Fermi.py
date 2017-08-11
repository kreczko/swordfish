#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import harpix as harp
import pylab as plt
import swordfish as sf
import pyfits as pf
from scipy.interpolate import interp1d
from tools import *
from copy import deepcopy

def get_los():

    # Define DM profile
    MW_D = 8.5 # kpc
    MW_rs = 20 # kpc
    alpha = 0.17
    MW_rhoS = 1.5*0.081351425781930664 # GeV cm^-3
    # Local density is 0.3 GeV/cm3 in this case
    kpc_cm = 3.086e21 # conversion factor
    def Lum_los(d, l, b):
        """Returns density squared for given galactic coordinates l and b at 
        distance d away from Suns location"""
        l = np.deg2rad(l)
        b = np.deg2rad(b)
        if (MW_D**2. + d**2. - (2*MW_D*d*np.cos(b)*np.cos(l))) < 0.0:
            R = 1e-5
        else:
            R = np.sqrt(MW_D**2. + d**2. - (2*MW_D*d*np.cos(b)*np.cos(l)))
        if R < 1e-5:
            R = 1e-5
        ratio = R/MW_rs
        # Einasto profile in units of GeV cm^-3
        #rho_dm = MW_rhoS*np.exp(-(2/alpha)*((ratio)**alpha - 1))
        rho_dm = MW_rhoS*np.exp(-(2/alpha)*((ratio)**alpha - 1))
        # Returns signal for annihilating DM rho**2
        return rho_dm**1.
    l = np.logspace(-3,np.log10(180),num=50)
    los = np.zeros(len(l))
    for i in range(len(l)):
        los[i] = quad(Lum_los,0.,30.,args=(l[i],0.0))[0]*kpc_cm
    Interp_sig = interp1d(l,los)
    return Interp_sig

def get_Jmap():
    Interp_sig = get_los()
    #sig = harp.HARPix(dims=dims).add_singularity((0,0), 1, 20, n = 10)
    Jmap = harp.HARPix().add_iso(16).add_disc((0,0), 30, 32)
    Jmap.add_func(lambda d: Interp_sig(d), mode = 'dist', center=(0,0))
    return Jmap

def plot_harp(h, filename):
    m = h.get_healpix(128)
    vmax = np.log10(m).max()
    vmin = vmax-2
    hp.mollview(np.log10(m), nest=True, min = vmin, max = vmax)
    plt.savefig(filename)

def get_BG():
    counts = hp.read_map("../data/1GeV_healpix_counts.fits")
    exposure  = hp.read_map("../data/1GeV_healpix_exposure.fits")
    bg = counts/exposure
    return bg, exposure

def MW_dSph():
    nside=32

    bg_hpx, exposure_hpx = get_BG()

    dims = ()

    J = get_Jmap()

    # Signal definition
    spec = 1.
    pos = (50, 40)
    dSph = harp.HARPix(dims = dims).add_singularity(pos, 0.1, 10, n = 10)
    #dSph = harp.HARPix(dims = dims).add_disc(pos, 10, 62).add_disc((50,-40), 10, 32)
    dSph.add_func(lambda d: 5e21/(.1+d)**2, mode = 'dist', center=pos)
    sig = J + dSph
    #sig.data += 1e21 # EGBG
    #plot_harp(sig, 'MW.eps')

    # Background definition
    #bg = harp.HARPix(dims = dims).add_iso(64)
    bg = harp.HARPix.from_healpix(bg_hpx, nest = False)
    exposure = harp.HARPix.from_healpix(exposure_hpx, nest = False)

    # Covariance matrix definition
    cov = harpix_Sigma(sig)
    bg_flat = deepcopy(bg)
    bg_flat.data *= 0
    bg_flat.data += 1e-9
    cov.add_systematics(err = bg_flat*0.1, sigma =  .25, Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  .5, Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  1., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  2., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  3., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  4., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma =  5., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma = 10., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma = 15., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma = 20., Sigma = None, nside =64)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 25., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 30., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 35., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 40., Sigma = None, nside = 1)

    # Set up swordfish
    fluxes = [sig.data.flatten()]
    noise = bg.get_formatted_like(sig).data.flatten()
    expmap = exposure.get_formatted_like(sig).data.flatten()*1e3
    systematics = cov
    m = sf.Swordfish(fluxes, noise, systematics, expmap, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    plot_harp(f, 'MW.eps')

def test():
    H = harp.HARPix().add_iso(4).add_disc((80,00), 40, 16).add_disc((-80,00), 40, 16)
    H.data += 1.

    # Covariance matrix definition
    cov = harpix_Sigma(H)
    flat = harp.HARPix().add_iso(16, fill=1.)
    cov.add_systematics(err = flat*.1, sigma = 3, Sigma = None, nside = 128)

    # Set up swordfish
    fluxes = [H.get_data(mul_sr=True)]
    noise = fluxes[0]
    expmap = noise*0. + 1e5
    systematics = cov
    m = sf.Swordfish(fluxes, noise, systematics, expmap, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(H, F, div_sr=True)
    print min(f.data), max(f.data)

    m = f.get_healpix(128)
    hp.mollview(m, nest=True)
    plt.savefig("MW.eps")

if __name__ == "__main__":
    test()
    #MW_dSph()
