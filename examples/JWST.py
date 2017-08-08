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


#################
# DM distribution
#################

def get_los(ADM = True):
    # Define DM profile
    MW_D = 8.5 # kpc
    MW_rs = 20 # kpc
    alpha = 0.17
    MW_rhoS = 0.081351425781930664 # GeV cm^-3
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
        # Returns signal for annihilating DM rho
        return rho_dm**(2 if ADM else 1)
    l = np.logspace(-3,np.log10(180),num=50)
    los = np.zeros(len(l))
    for i in range(len(l)):
        los[i] = quad(Lum_los,0.,100.,args=(l[i],0.0))[0]*kpc_cm
    Interp_sig = interp1d(l,los)
    return Interp_sig

def get_Jmap():
    Interp_sig = get_los()
    Jmap = harp.HARPix().add_disc((0,5.0), .1, 1024)
    Jmap.add_func(lambda d: Interp_sig(d), mode = 'dist', center=(0,0))
    return Jmap


#######################
# JWST characteristics
#######################

def get_instr_bkg(w):
    flux = lambda x: 1e12*(x/5)**4.0  # [mJy/sr] as function of wavelength [micron]
    # Wikipedia says:
    # 1 Jansky = 1e-26 Joule / Hz / s / m^2
    # 1 eV = 1.60218e-19 Joule
    # Watt = Joule / s
    # 1 eV = 241768.11129032 GHz
    # 1 eV corresponds to 1.24 micron
    # 1 micron corresponds to 299792.458 GHz
    # 1 Jy = 62.4 eV / GHz /s /m^2
    #
    # Jy/E_ph
    # l = c/nu
    # (l/micron)=(299792.458 GHz/nu)
    # dl/dnu = -micron*299792.48 GHz / nu^2
    # dl/dnu = -micron*299792.48 GHz / nu^2
    # dl/dnu = -l^2/c

    # Conversion
    # mJy/sr --> eV/GHz/s/sr/m2
    flux2 = lambda w: flux(w)*1e-6*1e-26/1.60218e-19*1e9
    E_ph = lambda w: 1.24/w  # [eV]
    # eV/GHz/s/sr/m2 --> ph/GHz/s/sr/m2
    flux3 = lambda w: flux2(w)/E_ph(w)
    # ph/GHz/s/sr/m2 --> ph/micron/s/sr/cm2
    c = 299792.e9  # micron/s
    flux4 = lambda w: flux3(w)*c/w**2*1e-4

    # (E/eV)*(w/1.24 micron) = 1
    # --> E = eV*(1.24 micron/w)
    #
    # Result: ph/sr/s/micron/cm2
    # 
    mu = w.integrate(flux4)
    return harp.HARPix().add_iso(1, fill=1.).expand(mu)

def get_Aeff(w):
    """Get effective area in cm2."""
    Aeff = lambda w: np.ones_like(w)*25.  # m^2
    eta = 0.10  # Efficiency
    return Aeff(w.means)*eta*1e4

def R(w):
   return np.ones_like(w)*0.005

def get_exposure(w, Tobs):
    Aeff = get_Aeff(w)
    obsT = Tobs*60.*60. # [s]
    return harp.HARPix().add_iso(1, fill = 1.).expand(Aeff*obsT)


##########
# DM model
##########

def get_sig_spec(tau, m, w, ch='bb'):
    w0 = 1.24/m  # eV --> micron
    sigma = 0.0001*w0
    spec_DM = lambda x: 1/np.sqrt(np.pi*2)/sigma*np.exp(-(x-w0)**2/2/sigma)
    return w.integrate(lambda x: 2*tau/4/np.pi/(m*1e-9)*spec_DM(x))


#######################
# Main routines
#######################

def JWST(m_DM, Tobs = 100.):
    # Parameters
    w = Logbins(np.log10(5), np.log10(30), 300)
    gamma0 = 1e-28  # 1/s

    # Get J-value map
    J = get_Jmap()

    # Define signal spectrum
    t = func_to_templates(lambda x, y: get_sig_spec(x*gamma0, y, w), [1., m_DM])

    # Get signal maps
    S = J.expand(t[0])

    # Get background (instr.)
    B = get_instr_bkg(w)

    # Get exposure
    expo = get_exposure(w, Tobs)
    fluxes, noise, systematics, exposure = get_model_input([S,], B, None, expo)
    m = Rockfish(fluxes, noise, systematics, exposure, solver='direct', verbose = False)

    # Calculate upper limits with effective counts method
    ec = EffectiveCounts(m)
    x_UL = ec.upperlimit(0.05, 0, gaussian = True)
    gamma_UL = x_UL*gamma0
    s, b = ec.effectivecounts(0, 1.)
    eV_J = 1.6e-19

    # Joule/s/m2
    flux_UL = (S).get_integral().sum()*x_UL*eV_J*m_DM*1e4/360

    print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
    print "Eff.  signal counts (theta = 1):", s
    print "Eff.  bkg counts (theta = 1)   :", b
    print "Upper limit on theta           :", gamma_UL
    print "Limit flux                     :", flux_UL


def toy():
    # at 10 micron
    w = 10  # micron
    nu = 3e8 / (1e-6*w)  # Hz
    I_bkg = 10.  # MJy/sr
    Aeff = 25.4  # m^2
    eta = 0.05  # efficiency
    R = 3000.  # spectral resolving power
    Omega = np.deg2rad(0.42/60./60.)**2*np.pi  # PSF size [sr]
    # Target: 10000 s obs time, 10 sigma detection, 8e-21 W/m2
    print Omega

    E_ph = 1.24/w  # eV
    F_bkg = 1e-20*I_bkg*Omega  # Jy = 1e-26*Joule/s/Hz/m2
    BW = nu/R

def toy2():
    w = 10  # micron
    E_ph = 1.24/w  # eV
    nu = 3e8 / (1e-6*w)  # Hz
    R = 300.  # spectral resolving power
    BW = nu/R
    mass = 2*E_ph
    tau = 1e27
    print "Mass [eV]:", mass
    print "Lifetime [s]:", tau
    J = get_los()(1.)
    Omega = np.pi*np.deg2rad(0.42/3600)**2
    F = E_ph/(mass*1e-9)/tau/4/np.pi*J*Omega  # eV/cm2/s
    eV_J = 1.6e-19
    Fc = F * 1e4 * eV_J  # Joule/s/m2
    print "Flux within Omega [W/m2]:", Fc
    I = Fc/BW/Omega/1e-20
    print "Differential intensity [MJy/sr]:", I

def toy3():
    Ja = get_los(ADM=True)(50)
    Jd = get_los(ADM=False)(1)
    print Jd
    Msol_GeV = 1.2e57
    kpc_cm = 3.086e21 # conversion factor
    Jnu = 1e8  # Msol/kpc
    print Jnu*Msol_GeV/kpc_cm**2
    print Jd/Ja

if __name__ == "__main__":
    #toy2()
    #JWST(0.1)
    toy3()
