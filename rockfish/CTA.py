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
import visual

def get_los():
    # Define DM profile
    MW_D = 8.5 # kpc
    MW_rs = 20 # kpc
    alpha = 0.17
    MW_rhoS = 0.081351425781930664 # GeV cm^-3
    m_DM = 1e3 # GeV
    kpc_cm = 3.086e21 # conversion factor
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
        #rho_dm = MW_rhoS*np.exp(-(2/alpha)*((ratio)**alpha - 1))
        rho_dm = MW_rhoS*np.exp(-(2/alpha)*((ratio)**alpha - 1))
        # Returns signal for annihilating DM rho**2
        return rho_dm**2.
    l = np.logspace(-3,np.log10(180),num=50)
    los = np.zeros(len(l))
    for i in range(len(l)):
        los[i] = quad(Lum_los,0.,100.,args=(l[i],0.0))[0]*kpc_cm
    Interp_sig = interp1d(l,los)
    return Interp_sig

def dNdE_e(E):
     # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
     E0 = 1e3
     return 1.17e-11*np.where(E > 1., (E/E0)**-3.9, (E/E0)**-3.0)

# Cosmic ray protons
def dNdE_p(E):
    # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
    # Factor of 3 comes from Silverwood paper and approximately matches
    # their plot
    E = E*3./1e3
    Norm = 8.73e-9
    Gamma = 2.71
    return Norm*(E**(-Gamma))

def CTA(m_DM, UL = True):

    # Define energy range
    E = Logbins(0.5, 4.5, 11) 
    dims = (E.num,)  # HARPix dimensions per pixel

    def get_sig(m_DM):
       # Get dN/dE integrated over energy bins (approx.)
       spec_DM = interp.Interp(ch='bb')
       spec_sig = E.integrate(lambda x: spec_DM(m_DM,x))

       # Get LOS integral
       Interp_sig = get_los()

       # HARPix signal definition
       sig = harp.HARPix(dims=dims).add_singularity((0,0), 1, 20, n = 10)
       #sig = harp.HARPix(dims=dims).add_disc((0,0), 10, 128)
       sig.add_func(lambda d: spec_sig*Interp_sig(d), mode = 'dist', center=(0,0))
       sig *= 1e-26/8/np.pi/(m_DM**2.)  # Reference cross-section
       return sig

    sig = get_sig(m_DM)
    dx = 0.1
    dsig = (get_sig(m_DM*(1.+dx))*(1.+dx)**-2+(sig*(-1.0)))*(dx*m_DM)**-1

#    plt.loglog(E.means, sig.get_integral()*E.means**1)
#    plt.loglog(E.means, abs(dsig.get_integral())*E.means**1)
#    plt.show()
#    quit()

    # Backgrounds
    # Should be in photons/cm^2/s/sr when multiplied by dE (approx)
    CR_Elec_bkg = E.integrate(dNdE_e)
    proton_eff = 1.e-2
    CR_Proton_bkg = proton_eff*E.integrate(dNdE_p)
    spec_bkg = CR_Proton_bkg + CR_Elec_bkg

    bkg = harp.HARPix(dims=dims).add_iso(1)
    bkg.add_func(lambda l, b: spec_bkg)

    # Covariance matrix for energy spectrum uncertainty (ad hoc)
    Sigma = get_sigma(E.means, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))

    # Set up the rockfish
    unc = 0.01 # 1% bkg uncertainty
    corr_length = 1  # 10 deg correlation length of bkg uncertainty

    # Effective area taken from https://portal.cta-observatory.org/CTA_Observatory/performance/SiteAssets/SitePages/Home/PPP-South-EffectiveAreaNoDirectionCut.png
    Et, EffA = np.loadtxt("CTA_effective_A.txt", unpack=True)
    EffectiveA_m2 = interp1d(Et,EffA, fill_value="extrapolate")(E.means)
    EffectiveA_cm2 = EffectiveA_m2*(100**2.) # Conversion to cm^2/50hr
    EffectiveA_cm2 = EffectiveA_cm2/50./60./60. # Conversion to cm^2/s
    EffectiveA_cm2 *= 0
    EffectiveA_cm2 += 1e10
    obsT = 0.1*60.*60. # 100 hours of observation in s
    expotab = obsT*EffectiveA_cm2  # Exposure in cm2 s  (Aeff * Tobs)
    expo = harp.HARPix(dims=dims).add_iso(1)
    expo.add_func(lambda l, b: expotab, mode = 'lonlat')

    if UL:
        fluxes, noise, systematics, exposure = get_model_input([sig], bkg,
                [dict(err = bkg*unc, sigma = corr_length, Sigma = Sigma, nside =
                   0)], expo)
        systematics = None
        m = Rockfish(fluxes, noise, systematics, exposure, solver='cg', verbose = True)

        # Calculate upper limits with effective counts method
        ec = EffectiveCounts(m)
        UL = ec.upperlimit(0.05, 0, gaussian = True)
        s, b = ec.effectivecounts(0, 1.)

        print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
        print "Eff.  signal counts (theta = 1):", s
        print "Eff.  bkg counts (theta = 1)   :", b
        print "Upper limit on theta           :", UL

        #F = m.effectiveinfoflux(0)
        #f = harp.HARPix.from_data(sig, F, div_sr = True)
        #m = f.get_healpix(512, idxs=(3,))
        #hp.mollview(m, nest = True)
        #plt.savefig('test.eps')
        return UL
    else:
        fluxes, noise, systematics, exposure = get_model_input([sig, dsig], bkg,
                [dict(err = bkg*unc, sigma = corr_length, Sigma = None, nside =
                   0)], expo)
        systematics = None
        m = Rockfish(fluxes, noise, systematics, exposure, solver='cg', verbose = False)
        F = m.fishermatrix()
        return F

def UL_plot():
    mlist = np.logspace(1, 3, 10)
    svlist = np.logspace(-26, -24, 10)
    sv0 = 1e-26
    ULlist = []
    G = np.zeros((len(mlist), len(svlist),2,2))
    for i, m in enumerate(mlist):
        UL = CTA(m, UL = True)
        ULlist.append(UL*sv0)
        F0 = CTA(m, UL = False)  # dtheta, dm, sv0 = 1e-26
        for j, sv in enumerate(svlist):
            F = F0.copy()

            # dt --> dsv, replace theta by sv
            F[0,0] /= sv0**2
            F[1,0] /= sv0
            F[0,1] /= sv0

            # sv0 --> sv, take into account larger flux
            F[1,1] *= (sv/sv0)**2
            F[0,1] *= sv/sv0
            F[1,0] *= sv/sv0

            # ln derivatives
            F[0,0] *= sv**2
            F[1,1] *= m**2
            F[0,1] *= sv*m
            F[1,0] *= sv*m

            # log10 derivatives
            F *= np.log10(np.e)**2

            G[i,j] = F

            # I_sv = I_t (dt/dsv)**2 = I_t / sv0**2
            # I_log10(sv) = I_sv (dlog10(sv)/dsv)**-2 
            #    = I_sv * sv**2 * (dlog10x/dlogx)**-2
            #    = I_sv * sv**2 * log10(e)**-2

    visual.fisherplot(np.log10(mlist), np.log10(svlist), G, xlog=True, ylog=True)
    plt.xlim([10, 1000])
    plt.ylim([1e-26, 1e-24])
    plt.loglog(mlist, ULlist)
    plt.savefig('test.eps')

if __name__ == "__main__":
    UL_plot()
