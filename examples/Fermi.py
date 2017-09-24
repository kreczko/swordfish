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

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = numpy.array(values)
    quantiles = numpy.array(quantiles)
    if sample_weight is None:
        sample_weight = numpy.ones(len(values))
    sample_weight = numpy.array(sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = numpy.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, values)

def get_extragalactic():
    data = hp.read_map("../data/extragalactic_jfactors_2GeVsmooth.fits")
    sr_per_pix = 4*np.pi/len(data)
    data /= sr_per_pix  # GeV^2 / cm^5
    J = harp.HARPix.from_healpix(data, nest = False)
    return J

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
        if R < 0.1e0:  # Using 1 kpc core
            R = 0.1e0
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
    Jmap = harp.HARPix().add_iso(32).add_disc((0,0), 20, 64)
    Jmap.add_func(lambda d: Interp_sig(d), mode = 'dist', center=(0,0))
    return Jmap

def plot_harp(h, filename, maxfac = 1):
    m = h.get_healpix(128)
    #vmax = np.log10(m).max()
    #vmin = vmax-3
    #hp.mollview(np.log10(m), nest=True, min = vmin, max = vmax)
    hp.mollview(m, nest=True, min = 0, max = maxfac*m.max())
    plt.savefig(filename)

def plot_perc(h, filename):
    m = h.get_healpix(128)

    hp.mollview(p)
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
    
    Jex = get_extragalactic()

    J = J + Jex

    # Signal definition
    spec = 1.
    dSph = harp.HARPix(dims = dims).add_iso(1, fill=1.)
    pos_list = [(50, 40), (-20, 30), (80, -80), (42, -15)]
    Jlist = [3e21, 3e21, 6e21, 2e21]
    for J0, pos in zip(Jlist, pos_list):
        dSph = dSph.add_singularity(pos, 0.1, 5, n = 10)
        #dSph = harp.HARPix(dims = dims).add_disc(pos, 10, 62).add_disc((50,-40), 10, 32)
        dSph.add_func(lambda d: J0/(.1+d)**2, mode = 'dist', center=pos)
    sig = J+dSph
    #sig.data += 1e21 # EGBG
    plot_harp(sig, 'sig.eps', maxfac = 1.0)

    # Background definition
    #bg = harp.HARPix(dims = dims).add_iso(64)
    bg = harp.HARPix.from_healpix(bg_hpx, nest = False)
    plot_harp(bg, 'bg.eps', maxfac = 0.01)

    exposure = harp.HARPix.from_healpix(exposure_hpx, nest = False)

    # Covariance matrix definition
    cov = harpix_Sigma(sig)
    bg_flat = deepcopy(bg)
    bg_flat.data *= 0
    bg_flat.data += 1e-8
#    cov.add_systematics(err = bg_flat*0.1, sigma =  .25, Sigma = None, nside =64)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  .5, Sigma = None, nside =16)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  1., Sigma = None, nside =16)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  2., Sigma = None, nside =64)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  3., Sigma = None, nside =64)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  4., Sigma = None, nside =64)
#    cov.add_systematics(err = bg_flat*0.1, sigma =  7., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.1, sigma = 10., Sigma = None, nside =64)
    cov.add_systematics(err = bg_flat*0.3, sigma = 1e10, Sigma = None, nside =8)
    cov.add_systematics(err = bg_flat*0.1, sigma = 20., Sigma = None, nside =64)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 25., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 30., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 35., Sigma = None, nside = 1)
    #cov.add_systematics(err = bg_flat*0.1, sigma = 40., Sigma = None, nside = 1)

    # Set up swordfish
    fluxes = [sig.get_data(mul_sr=True)]
    noise = bg.get_formatted_like(sig).get_data(mul_sr=True)
    expmap = exposure.get_formatted_like(sig).get_data()*1e-4
    systematics = cov
    #systematics = None
    m = sf.Swordfish(fluxes, noise, systematics, expmap, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F, div_sr=True)
    plot_harp(f, 'MW.eps', maxfac = 0.1)

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
    #test()
    MW_dSph()
    #get_extragalactic()
