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
import visual

def DD(m_DM, UL = True):
    rho_0 = 0.3*1.e3 # MeV cm^-3
    xi_T = 10.# mass fraction
    g = 1.
    # F_T = 10.
    m_T = 68.*1.e3 # MeV Germanuin example 
    m_med = 10. # MeV

    # Define energy range
    E = Logbins(0.5, 4.5, 11) 

    def eta_F():
        v, gave = np.loadtxt("DD_files/gave.dat", unpack=True, dtype=float)
        f = interp1d(v, gave) # 1/(km/s)
        return f

    def dRdE(E_R):
        # g is the parameter we wish to set a limit on so is left out
        # Form factor taken from eq4.4 of http://pa.brown.edu/articles/Lewin_Smith_DM_Review.pdf
        # FIXME: Should correct for realistic form factor
        F_T = lambda E_R: np.sqrt(np.exp(-(1./3.)*(E_R**2.)))
        mu_T = m_DM*m_T/(m_DM+m_T)
        vmin = lambda E_R: np.sqrt(m_T*E_R/2/(mu_T**2.))
        eta = eta_F()
        sig = rho_0*xi_T*(F_T(E_R)**2.)*eta(vmin(E_R))/2./np.pi/m_DM/((2*m_T*E_R + m_med**2.)**2.)
        return sig

    # sig = get_sig(m_DM, E_R)
    plt.loglog(E.means, dRdE(E.means))
    # plt.loglog(E.means, abs(dsig.get_integral())*E.means**1)
    plt.show()
    quit()

################################### Background definitions

    # Backgrounds
    # For CREST-II we assume there are no bakcground events

    # Covariance matrix for energy spectrum uncertainty (ad hoc)
    Sigma = get_sigma(E.means, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))

    # Set up the swordfish
    unc = 0.01 # 1% bkg uncertainty
    corr_length = 1  # 10 deg correlation length of bkg uncertainty

################################### Exposure Goes here
    obsT = 0.1*60.*60. # 100 hours of observation in s

    if UL:
        fluxes, noise, systematics, exposure = get_model_input([sig], bkg,
                [dict(err = bkg*unc, sigma = corr_length, Sigma = Sigma, nside =
                   0)], expo)
        systematics = None
        m = Model(fluxes, noise, systematics, exposure, solver='cg', verbose = True)

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
        m = Model(fluxes, noise, systematics, exposure, solver='cg', verbose = False)
        F = m.fishermatrix()
        return F


if __name__ == "__main__":
    DD(100*1.e3)
