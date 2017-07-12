#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import HARPix as harp
import pylab as plt
from core import *
from tools import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
from math import cos, sin
from PPPC4DMID import interp

def CTA_UL():
    # Define energy range
    E = np.logspace(0.5, 4.5, 11)  # 10 energy bins
    Ebins = np.array(zip(E[:-1], E[1:]))  # energy bins
    dE = Ebins[:,1]-Ebins[:,0]  # bin width
    Emed = Ebins.prod(axis=1)**0.5  # geometric mean enery
    dims = (len(Emed),)  # HARPix dimensions per pixel

    # Define base pixel size
    nside = 8

    MW_D = 8.5 # kpc
    MW_rs = 20 # kpc
    alpha = 0.17
    MW_rhoS = 0.081351425781930664 # GeV cm^-3
    m_DM = 1e3 # GeV

    def Lum_los(d, l, b):
        """Returns density squared for given galactic coordinates l and b at 
        distance d away from Suns location"""
        l = np.deg2rad(l)
        b = np.deg2rad(b)
        if (MW_D**2. + d**2. - (2*MW_D*d*cos(b)*cos(l))) < 0.0:
            R = 1e-5
        else:
            R = np.sqrt(MW_D**2. + d**2. - (2*MW_D*d*cos(b)*cos(l)))
        if R < 1e-5:
            R = 1e-5
        ratio = R/MW_rs
        # Einasto profile in units of GeV cm^-3
        rho_dm = MW_rhoS*np.exp(-(2/alpha)*((ratio)**alpha - 1))
        # Returns signal for annihilating DM rho**2
        return rho_dm**2.

    l = np.logspace(-3,4,num=50)
    los = np.zeros(len(l))
    for i in range(len(l)):
        los[i] = quad(Lum_los,0.,200.,args=(l[i],0.0))[0]

    Interp_sig = interp1d(l,los)
    # Signal definition
    # DEFINE SIGNAL SPCTRUM HERE FROM PPPC4DMID
    spec_DM = interp.Interp(ch='bb')
    spec_sig = spec_DM(m_DM,Emed)*dE # dN/dE integrated over energy bins (approx.)
    # Can put in additional dimensions in harp.HARPix(dims=dims)
    sig = harp.HARPix(dims=dims).add_singularity((0,0), 2, 15, n = 1000)
    sig.add_func(lambda d: spec_sig*Interp_sig(d), mode = 'dist', center=(0,0))
    # DM prefactors, we deive limit on <sigmav>
    sig *= 8/np.pi/(m_DM**2.)

    # NOTE: sig.data should correpond to the signal intensity (for a fixed DM
    # mass and annihilation cross-section), in units of photons/cm2/s/sr,
    # integrated over the energy bin, but *not* integrated over the pixel size
    
    # hp.mollview(np.log10(sig.get_healpix(128)), nest=True)
    # plt.show()
    # plot_harp(f, 'CTA_test.eps')
    # quit()

    # Cosmic ray electrons
    def CR_Elec_spec(E):
        # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
        E = E/1.e3
        Norm = 1.17e-11
        if E > 1.:
            gamma = 3.9
        else:
            gamma = 3.0
        return Norm*(E**(-gamma))

    # Should be in photons/cm^2/s/sr when multiplied by dE (approx)
    CR_Elec_bkg = (np.vectorize(CR_Elec_spec))(Emed)*dE

   # Cosmic ray protons
    def CR_Proton_spec(E):
        # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
        # Factor of 3 comes from Silverwood paper and approximately matches
        # their plot
        E = E*3./1e3
        Norm = 8.73e-9
        Gamma = 2.71
        return Norm*(E**(-Gamma))

    proton_eff = 1.e-2
    # Should be in photons/cm^2/s/sr when multiplied by dE (approx)
    CR_Proton_bkg = proton_eff*(np.vectorize(CR_Proton_spec))(Emed)*dE
    
    #### Might not be necessary
    spec_bkg = CR_Proton_bkg + CR_Elec_bkg
    ################

    # NOTE: Check plot with Fig2: 1408.4131v2
    # FIXME: DM signal much too low
    # Jval = sig.get_healpix(128).sum()
    # Jval *= 
    # plt.loglog(Emed, Emed**2.*spec_bkg, label="Total Background")
    # plt.loglog(Emed, Emed**2.*CR_Proton_bkg, label="CR protons")
    # plt.loglog(Emed, Emed**2.*CR_Elec_bkg, label="CR electrons")
    # plt.loglog(Emed, Emed**2.*spec_sig*Jval, label="bb DM")
    # plt.xlim(10,1e4)
    # plt.xlabel("Energy")
    # plt.ylabel("Differential intensity")
    # plt.legend()
    # plt.show()
    # quit()
    # Background definition
    # spec_bkg = CR_Elec_bkg + CR_Proton_bkg  # dN/dE integrated over energy bins (approx.)

    bkg = harp.zeros_like(sig)
    bkg.add_func(lambda l, b: spec_bkg)
    # hp.mollview(np.log10(bkg.get_healpix(128)), nest=True)
    # plt.show()
    # quit()

    # NOTE: Same as for the signal.

    # Covariance matrix for energy spectrum uncertainty (ad hoc)
    # Sigma = get_sigma(Emed, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))

    # Set up rockfish
    unc = 0.01  # 1% bkg uncertainty
    corr_length = 10  # 10 deg correlation length of bkg uncertainty

    # Effective area taken from https://portal.cta-observatory.org/CTA_Observatory/performance/SiteAssets/SitePages/Home/PPP-South-EffectiveAreaNoDirectionCut.png
    Et, EffA = np.loadtxt("CTA_effective_A.txt", unpack=True)
    EffectiveA_m2 = interp1d(Et,EffA, fill_value="extrapolate")(Emed)
    EffectiveA_cm2 = EffectiveA_m2/(100**2.) # Conversion to cm^2/50hr
    EffectiveA_cm2 = EffectiveA_cm2/50./60./60. # Conversion to cm^2/s
    obsT = 100.*60.*60. # 100 hours of observation in s
    expo = obsT*EffectiveA_cm2  # Exposure in cm2 s  (Aeff * Tobs)
    # print expo.sum()
    # NOTE: expo can be HARPix object, and should be energy dependent in the
    # end

    # FIXME: Currently does not take in exposure map as a function of energy so just sums instead
    fluxes, noise, systematics, exposure = get_model_input([sig], bkg,
            [dict(err = bkg*unc, sigmas = [corr_length], Sigma = None, nside = nside)], expo.sum())
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
