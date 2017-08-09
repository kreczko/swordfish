#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import harpix as harp
import pylab as plt
import swordfish as sf
from tools import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
from math import cos, sin
from PPPC4DMID import interp
import metricplot as mp
import DSspectra

#################
# DM distribution
#################

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
        los[i] = quad(Lum_los,0.,30.,args=(l[i],0.0))[0]*kpc_cm
    Interp_sig = interp1d(l,los)
    return Interp_sig

def get_Jmap():
   Interp_sig = get_los()
   #sig = harp.HARPix(dims=dims).add_singularity((0,0), 1, 20, n = 10)
   Jmap = harp.HARPix().add_disc((0,0), 5, 32)
   Jmap.add_func(lambda d: Interp_sig(d), mode = 'dist', center=(0,0))
   return Jmap


#######################
# CTA characteristics
#######################

def dNdE_e(E):
     # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
     E0 = 1e3
     return 1.17e-11*np.where(E > 1e3, (E/E0)**-3.9, (E/E0)**-3.0)

# Cosmic ray protons
def dNdE_p(E):
    # dPhi/dE/dOmega in (GeV cm^2 s sr)^-1
    # Factor of 3 comes from Silverwood paper and approximately matches
    # their plot
    E = E*3./1e3
    Norm = 8.73e-9
    Gamma = 2.71
    return Norm*(E**(-Gamma))

def get_instr_bkg(E):
    CR_Elec_bkg = E.integrate(dNdE_e)
    proton_eff = 1.e-2
    CR_Proton_bkg = proton_eff*E.integrate(dNdE_p)
    spec_bkg = CR_Proton_bkg + CR_Elec_bkg
    return harp.HARPix().add_iso(1, fill=1.).expand(spec_bkg)

def get_exposure(E, Tobs):
    # Effective area taken from https://portal.cta-observatory.org/CTA_Observatory/performance/SiteAssets/SitePages/Home/PPP-South-EffectiveAreaNoDirectionCut.png
    Et, EffA = np.loadtxt("../data/CTA_effective_A.txt", unpack=True)
    Et *= 1e3  # TeV --> GeV
    EffA *= 1e4  # m2 --> cm2
    EffectiveA_cm2 = interp1d(Et, EffA, fill_value="extrapolate")(E.means)
    obsT = Tobs*3600 # 100 hours of observation in s
    expotab = obsT*EffectiveA_cm2  # Exposure in cm2 s  (Aeff * Tobs)
    return harp.HARPix().add_iso(1, fill = 1.).expand(expotab)


##########
# DM model
##########


def get_sig_spec(sv, m, E, ch='bb'):
    #spec_DM = interp.Interp(ch=ch)
    #return E.integrate(lambda x: sv/8/np.pi/m**2*spec_DM(m,x))
    s = DSspectra.spec(channel='bb', mass = m, type='gam')
    return E.integrate(lambda x: sv/8/np.pi/m**2*s(x))


#######################
# Main routines
#######################


def CTA(m_DM, UL = True, syst_flag = True, Tobs = 100.):
    # Parameters
    E = Logbins(1.0, 4.0, 50)   # GeV 10 GeV - 10 TeV
    unc = 0.01 # 1% bkg uncertainty
    corr_length = 1  # 10 deg correlation length of bkg uncertainty
    Sigma = get_sigma(E.means, lambda x, y: np.exp(-(x-y)**2/2/(x*y)/0.5**2))
    sv0 = 1e-26

    # Get J-value map
    J = get_Jmap()

    # Define signal spectrum
    t = sf.func_to_templates(lambda x, y: get_sig_spec(x*sv0, y, E), [1.,
       m_DM], dx = [.01,m_DM*0.01])

    # Get signal maps
    S = J.expand(t[0])
    dS = J.expand(t[1])

    # Get background (instr.)
    B = get_instr_bkg(E)  # FIXME?

#    plt.loglog(E.means, B.get_integral()/4./np.pi)
#    plt.loglog(E.means, S.get_integral()/4./np.pi)
#    plt.show()
#    quit()

    # Get exposure
    expo = get_exposure(E, Tobs)
    flux = [S,] if UL else [dS,S]
    fluxes, noise, systematics, exposure = get_model_input(flux, B,
            [dict(err = B*unc, sigma = corr_length, Sigma = Sigma, nside = 0)], expo)
    if not syst_flag: systematics = None
    m = sf.Swordfish(fluxes, noise, systematics, exposure, solver='direct', verbose = False)

    if UL:
        # Calculate upper limits with effective counts method
        ec = sf.EffectiveCounts(m)
        x_UL = ec.upperlimit(0.05, 0, gaussian = True)
        sv_UL = x_UL*sv0
        s, b = ec.effectivecounts(0, 1.)
#        print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
#        print "Eff.  signal counts (theta = 1):", s
#        print "Eff.  bkg counts (theta = 1)   :", b
#        print "Upper limit on theta           :", sv_UL

        return sv_UL
    else:
        F = m.fishermatrix()  # w.r.t. (x,m)
        F[1,1] /= sv0**2
        F[0,1] /= sv0
        F[1,0] /= sv0
        return F

def generate_dump(syst_flag = True):
    mlist = np.logspace(1.5, 4.0, 10)
    svlist = np.logspace(-27, -24, 31)
    ULlist = []
    G = np.zeros((len(svlist), len(mlist),2,2))
    for i, m in enumerate(mlist):
        UL = CTA(m, UL = True, syst_flag = syst_flag)
        print "Mass [GeV]", m
        print "Upper limit [cm3/s]:", UL
        ULlist.append(UL)
        sv0 = 1e-26
        F0 = CTA(m, UL = False, syst_flag = syst_flag)  # dtheta, dm, sv0 = 1e-26
        for j, sv in enumerate(svlist):
            F = F0.copy()

            # sv0 --> sv, take into account larger flux
            F[0,0] *= (sv/sv0)**2
            F[0,1] *= sv/sv0
            F[1,0] *= sv/sv0

            G[j,i] = F

    np.savez('dump1.npz', x=mlist, y=svlist, g=G)
    np.savez('dump2.npz', x=mlist, y=ULlist)

def CTA_plot():
    #visual.loglog_from_npz('dump.npz')
    tf = mp.TensorField.fromfile('dump1.npz', logx = True, logy = True)
    #tf.quiver()
    #plt.savefig('test.eps')
    #quit()
    vf1, vf2 = tf.get_VectorFields()
    mask = lambda x, y: y < np.log10(3e-26)
    lines = vf1.get_streamlines([2, -26.0], Nmax=100, mask = mask, Nsteps = 100)
    for line in lines:
       line = 10**line
       plt.plot(line.T[0], line.T[1], color='0.5')
    lines = vf2.get_streamlines([2, -26.0], Nmax=100, mask = mask, Nsteps = 100)
    for line in lines:
       line = 10**line
       plt.plot(line.T[0], line.T[1], color='0.5')

    contour = 10**tf.get_contour([3, -26.0], 1, Npoints = 300)
    plt.plot(contour.T[0], contour.T[1], 'b')
    contour = 10**tf.get_contour([3, -26.0], 2, Npoints = 300)
    plt.plot(contour.T[0], contour.T[1], 'b--')
    contour = 10**tf.get_contour([2, -26.0], 1, Npoints = 300)
    plt.plot(contour.T[0], contour.T[1], 'b')
    contour = 10**tf.get_contour([2, -26.0], 2, Npoints = 300)
    plt.plot(contour.T[0], contour.T[1], 'b--')

    x = np.load('dump2.npz')['x']
    y = np.load('dump2.npz')['y']
    plt.loglog(x, y, 'r')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.savefig('test.eps')

if __name__ == "__main__":
    generate_dump(syst_flag = False)
    CTA_plot()
    #CTA(100, UL = True, syst_flag = True)
    #print CTA(100., UL = True, syst_flag = True, Tobs = .0001)
    #print CTA(100., UL = True, syst_flag = True, Tobs = .01)
    #print CTA(100., UL = True, syst_flag = False, Tobs = 100.)
    #print CTA(400., UL = True, syst_flag = True, Tobs = 100.)
    #print CTA(100., UL = True, syst_flag = True, Tobs = 10000.)
    #print CTA(100., UL = True, syst_flag = True, Tobs = 1000000.)
    #print CTA(100., UL = True, syst_flag = True, Tobs = 100000000.)
