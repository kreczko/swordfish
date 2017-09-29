#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import pylab as plt
import swordfish as sf
from scipy.interpolate import interp1d
from scipy.integrate import quad
from math import cos, sin
from scipy.special import erf
from scipy.constants import c
import visual

g2 = 1.e-18

def DD(m_DM, UL = True):
    """ This routine will try to recreate the limits from 1708.08571 for the CRESST-III Experiment, a future version of CRESST"""
    c_kms = c*1.e-3 # in km s^-1
    GeV_kg = 1.8e-27
    sGeV = 1/6.58e-25
    mGeV = 1/0.197e-15
    kgs = sGeV/GeV_kg  # = kg * s in natural units
    h = 4.135667662e-15 # eV s
    h_MeVs = h/1.e6
    m_DM *= 1.e3 # conversion to MeV
    rho_0 = 0.3  # GeV/cm3
    rho_0 *= 1e3*1e2**3/mGeV**3*1e3**3  # GeV/cm3 --> MeV^4
    xi_T = 1. # FIXME: Need real values
    m_T = 68.*1.e3 # MeV FIXME: Need real values, currently set to Germanium
    m_med = 10. # MeV
    muT = m_DM*m_T/(m_DM + m_T)

    # Define energy range in MeV
    E = np.linspace(0.1*1e-3, 2.*1e-3, num=19) 
    Ewidth = E[1]-E[0]
    Emeans = E[0:-1]+Ewidth/2.

    def eta_F():
        v, gave = np.loadtxt("DD_files/gave.dat", unpack=True, dtype=float)
        f = interp1d(v, gave) # Natural units
        return f

    def dRdE(E_R):
        """Return differential recoil rate in 1/s/keV/kg."""
        # g is the parameter we wish to set a limit on so is left out
        # Form factor taken from eq4.4 of http://pa.brown.edu/articles/Lewin_Smith_DM_Review.pdf
        # TODO: Z,A,theta dependence missing
        # FIXME: Should correct for realistic form factor
        F_T = lambda E_R: np.sqrt(np.exp(-(1./3.)*(E_R**2.)))
        vmin = lambda E_R: np.sqrt(m_T*E_R/2/(muT**2.))
        eta = eta_F()
        signal = rho_0*xi_T*g2*F_T(E_R)**2*eta(vmin(E_R))/2./np.pi/m_DM/((2*m_T*E_R+m_med**2)**2)
        signal *= kgs  # 1/MeV -->  1/s/MeV/kg
        return signal

    def ScatterProb(q, E1, E2):
        # CRESST definitions
        # Energy resoultion is set by a Gaussian at 20 eV
        # FIXME: Check energy resolution implementation
        # var = lambda x: np.exp(((x-muCRESST)**2.)/2./(sigmaCRESST**2))/2./np.pi
        E_R = q**2./2./m_T
        # var = 20.*1.e-6 # MeV
        var = 1. # MeV
        # print erf((E1-E_R)/np.sqrt(2)/var)
        prob = (erf((E2-E_R)/np.sqrt(2)/var) - erf((E1-E_R)/np.sqrt(2)/var))/2.
        # prob = 1.
        # print "Probability", prob
        return prob

    Eth = 100*1e-6 # threshold energy [MeV]
    Vesc = 544./c_kms
    Vobs = 232./c_kms
    qmin = np.sqrt(2.*m_T*Eth)  # MeV
    qmax = 2.*muT*(Vesc + Vobs)  # MeV
    E_Rmin = qmin**2./2./m_T
    E_Rmax = qmax**2./2./m_T

    if qmin > qmax:
        print "Skip m_DM [GeV]", m_DM*1e-3
        return None

#    sig = np.zeros(len(Emeans))
#    sig_dif = lambda ER, x1, x2: ScatterProb(ER, x1, x2)*dRdE(ER)
#
#    for i in range(len(Emeans)):
#        sig[i] = quad(sig_dif, E_Rmin, E_Rmax, args=(E[i],E[i+1]))[0]

    sig = dRdE(Emeans)*Ewidth

    # print dRdE_A, qdiff
    # print Emeans, sig
    # plt.loglog(Emeans, sig)
    # plt.xlabel(r'$E_R (MeV)$')
    # plt.ylabel(r'$dR/dE_R$')
    # plt.show()
    # quit()

################################### Background definitions

    # Backgrounds
    # For CREST-III we assume 1000kg days of exposure with energy threshold of 100eV. There are 19 bins between 0.1 and 2 keV

    # Background level assumed to be 3.5e-2 keV^-1 kg^-1 day^-1
    #bkg = np.zeros(len(E))
    #bkg += 3.5
    bkg = np.ones_like(sig)*3.5e-2*(Ewidth*1e3)/(3600.*24)

    # Covariance matrix for energy spectrum uncertainty (ad hoc)
    # Sigma = get_sigma(Emeans, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))

    # Set up the swordfish
    unc = 0.01 # 1% bkg uncertainty
    corr_length = 1  # 10 deg correlation length of bkg uncertainty

################################### Exposure Goes here
    obsT = np.ones_like(sig)*100*3600 # 100 hours of observation in s

    if UL:
        systematics = None
        m = sf.Swordfish(sig, bkg, systematics, obsT, solver='cg', verbose = True)

        # Calculate upper limits with effective counts method
        ec = sf.EffectiveCounts(m)
        UL = ec.upperlimit(0.05, 0, gaussian = True)
        s, b = ec.effectivecounts(0, 1.)

        print "DM mass [GeV]:", m_DM*1e-3
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
        # fluxes, noise, systematics, exposure = get_model_input([sig, dsig], bkg,
        #         [dict(err = bkg*unc, sigma = corr_length, Sigma = None, nside =
        #            0)], expo)
        # systematics = None
        m = sf.Swordfish(fluxes, noise, systematics, exposure, solver='cg', verbose = False)
        F = m.fishermatrix()
        return F

def UL_plot():
    mDM, glim = np.loadtxt("glim.txt", unpack=True, dtype=float)
    mlist = np.logspace(0.0, 1, 10)
    ULlist = []
    for i, m in enumerate(mlist):
        UL = DD(m, UL = True)
        if UL is None:
            ULlist.append(0.)
        else:
            ULlist.append(UL*g2)

    plt.loglog(mlist, np.sqrt(ULlist), label="My calculation")
    plt.loglog(mDM, glim, label="From paper")
    # plt.legend()
    plt.show()
    quit()

if __name__ == "__main__":
    UL_plot()
