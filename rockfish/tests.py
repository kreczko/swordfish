#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import HARPix as harp
import pylab as plt
from core import *
from tools import *

def test_3d():
    def plot_harp(h, filename, dims = ()):
        m = h.get_healpix(128, idxs= dims)
        hp.mollview(m, nest=True)
        plt.savefig(filename)

    def plot_rock(h, filename):
        nside = 32
        N = h.dims[0]
        npix = hp.nside2npix(nside)
        rings = hp._pixelfunc.pix2ring(nside, np.arange(npix), nest = True)
        T = harp.get_trans_matrix(h, nside)
        data = harp.trans_data(T, h.data)
        out = []
        for r in range(1, 4*nside):
            mask = rings == r
            out.append(data[mask].sum(axis=0))
        out = np.array(out)
        plt.imshow(out, aspect=0.8*N/(4*nside))
        plt.xlabel("Energy [AU]")
        plt.ylabel("Latitude [AU]")
        plt.savefig(filename)

    x = np.linspace(0, 10, 20)
    nside = 16

    dims = (len(x),)

    # Signal definition
    spec_sig = np.exp(-(x-5)**2/2)
    sig = harp.HARPix(dims = dims).add_iso(nside).add_singularity( (50,50), 1, 20, n = 10)
    sig.add_func(lambda d: spec_sig*np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.add_func(lambda d: spec_sig/(d+1)**1, mode = 'dist', center=(50,50))
    sig.mul_sr()

    # Background definition
    bg = harp.zeros_like(sig)
    spec_bg = x*0. + 1.
    bg.add_func(lambda l, b: spec_bg*(0./(b**2+1.)**0.5+0.1))
    bg.mul_sr()

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    corr = lambda x, y: np.exp(-(x-y)**2/2/3**2)
    Sigma = get_sigma(x, corr)
    cov.add_systematics(err = bg*0.1, sigmas = [20.,], Sigma = Sigma, nside = 16)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*100.0
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    f.div_sr()
    plot_harp(f, 'test.eps', dims = (11,))
    plot_rock(f, 'test.eps')


def test_UL():
    # Signal definition
    sig = harp.HARPix().add_iso(16)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))

    # Background definition
    bg = harp.zeros_like(sig)
    bg.add_func(lambda l, b: 0./(b**2+1.)**0.5+1.0)

    fluxes, noise, systematics, exposure = get_model_input(
            [sig], bg, [dict(err=bg*0.1, sigmas = [20.,], Sigma = None, nside =
                16)], bg*100.)
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')

    I = m.fishermatrix()
    print I

    ec = EffectiveCounts(m)
    UL = ec.upperlimit(0.05, 0)
    ULg = ec.upperlimit(0.05, 0, gaussian = True)
    s, b = ec.effectivecounts(0, 1.0)

    print "Total signal counts (theta = 1):", ec.counts(0, 1.0)
    print "Eff.  signal counts (theta = 1):", s
    print "Eff.  bkg counts (theta = 1)   :", b
    print "Upper limit on theta           :", UL
    print "Upper limit on theta (gaussian):", ULg

def test_simple():
    def plot_harp(h, filename):
        m = h.get_healpix(128)
        hp.mollview(m, nest=True)
        plt.savefig(filename)

    dims = ()
    nside = 16

    # Signal definition
    spec = 1.
    sig = harp.HARPix(dims = dims).add_iso(nside).add_singularity( (50,50), .1, 20, n = 100)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.add_func(lambda d: 1/(d+1)**1, mode = 'dist', center=(50,50))
    sig.data += 0.1  # EGBG
    plot_harp(sig, 'sig.eps')
    sig.mul_sr()

    # Background definition
    bg = harp.zeros_like(sig)
    bg.add_func(lambda l, b: 0./(b**2+1.)**0.5+1.0)
    plot_harp(bg, 'bg.eps')
    bg.mul_sr()
    #bg.print_info()

    # Covariance matrix definition

    cov = HARPix_Sigma(sig)
    cov.add_systematics(err = bg*0.1, sigmas = [20.,], Sigma = None, nside = 64)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*10000.
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0, thetas = [0.000], psi = 1.)
    f = harp.HARPix.from_data(sig, F)
    f.div_sr()
    plot_harp(f, 'test.eps')
    quit()

def test_MW_dSph():
    def plot_harp(h, filename):
        m = h.get_healpix(128)
        hp.mollview(np.log10(m), nest=True)
        plt.savefig(filename)

    dims = ()

    # Signal definition
    spec = 1.
    MW = harp.HARPix(dims = dims).add_iso(8).add_singularity((0,0), 0.1, 20, n = 100)
    MW.add_func(lambda d: spec/(.1+d)**2, mode = 'dist', center=(0,0))
    pos = (50, 40)
    dSph = harp.HARPix(dims = dims).add_singularity(pos, 0.1, 20, n = 100)
    dSph.add_func(lambda d: 0.1*spec/(.1+d)**2, mode = 'dist', center=pos)
    sig = MW + dSph
    sig.data += 1  # EGBG
    plot_harp(sig, 'sig.eps')

    # Background definition
    bg = harp.HARPix(dims = dims).add_iso(64)
    bg.add_func(lambda l, b: 1/(b+1)**2)
    plot_harp(bg, 'bg.eps')

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    var = bg*bg
    var.data *= 0.1  # 10% uncertainty
    cov.add_systematics(variance = var, sigmas = [100], Sigma = None, nside =
            nside)

    # Set up rockfish
    fluxes = [sig.data.flatten()]
    noise = bg.get_formatted_like(sig).data.flatten()
    systematics = cov
    exposure = np.ones_like(noise)*1.
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')

    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F)
    plot_harp(f, 'test.eps')

def test_spectra():
    x = np.linspace(0, 10, 1000)  # energy
    dx = x[1]-x[0]  # bin size

    fluxes = [(np.exp(-(x-5)**2/2/3**2)+np.exp(-(x-5)**2/2/0.2**2))*dx]
    noise = (1+x*0.0001)*dx
    exposure = np.ones_like(noise)*1.
    X, Y = np.meshgrid(x,x)
    systematics = 0.1*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/40**2) + np.exp(-(X-Y)**2/2/20**2)
                )).dot(np.diag(noise))
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')
    f = m.effectiveinfoflux(0)
    plt.plot(np.sqrt(fluxes[0]**2/dx/noise))
    plt.plot(np.sqrt(f/dx), label='Info flux')
    plt.legend()
    plt.savefig('test.eps')

def smoothtest():
    nside = 64
    sigma = 30

    lmax = 3*nside - 1  # default from hp.smoothing
    lmax += 10
    Nalm = hp.Alm.getsize(lmax)
    H = hp.smoothalm(np.ones(Nalm), sigma = np.deg2rad(sigma), inplace = False)
    npix = hp.nside2npix(nside)
    m = np.zeros(npix)
    m[1000] = 1
    M = hp.alm2map(hp.map2alm(m, lmax = lmax)*H, nside, lmax = lmax)
    G = H/max(M)
    m *= 0
    m[0] = 1
    I = hp.alm2map(hp.map2alm(m, lmax = lmax)*G, nside, lmax = lmax)
    print max(I)
    hp.mollview(I)
    plt.savefig('test.eps')

def test_covariance():
    nside = 8

    # Signal definition
    sig = harp.HARPix().add_iso(nside).add_singularity( (50,50), .1, 50, n = 100)
    sig.add_func(lambda d: np.exp(-d**2/2/20**2), mode = 'dist', center=(0,0))
    sig.mul_sr()

    # Covariance matrix definition
    cov = HARPix_Sigma(sig)
    bg = sig*.0
    bg.data += 1
    cov.add_systematics(err = bg*0.1,
            sigmas = [20.,], Sigma = None, nside = nside)

    x = np.zeros_like(sig.data)
    x[901] = 1.
    y = cov.dot(x)
    sig.data = y
    sig.div_sr()
    z = sig.get_healpix(128)
    hp.mollview(z, nest = True)
    plt.savefig('test.eps')

def test_matrix():
    x = np.linspace(0, 10, 1000)  # energy
    dx = x[1]-x[0]  # bin size

    fluxes = [np.exp(-(x-5)**2/2/3**2)*dx,np.exp(-(x-5)**2/2/0.2**2)*dx]
    noise = np.ones_like(x)*dx
    exposure = np.ones_like(noise)*1.001
    X, Y = np.meshgrid(x,x)
    systematics = 0.1*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/40**2) + np.exp(-(X-Y)**2/2/20**2)
                )).dot(np.diag(noise))
    m = Rockfish(fluxes, noise, systematics, exposure, solver='cg')
    I = m.fishermatrix()
    print I
    F = m.infoflux()
    f = m.effectiveinfoflux(0)
    plt.plot(x, f*dx, label='Feff')
    plt.plot(x, F[0,0]*dx, label='F00')
    plt.plot(x, F[1,1]*dx, label='F11')
    plt.plot(x, F[0,1]*dx, label='F01')
    plt.legend()
    plt.savefig('test.eps')

def test_infoflux():
    sig = harp.HARPix().add_singularity( (0,0), .1, 50, n = 100)
    sig.data += 1.
    bkg = sig

    fluxes, noise, systematics, exposure = get_model_input([sig], bkg, None, 1.)
    m = Rockfish(fluxes, noise, systematics, exposure)
    F = m.effectiveinfoflux(0)
    f = harp.HARPix.from_data(sig, F, div_sr = True)
    m = f.get_healpix(256)
    hp.mollview(m, nest = True)
    plt.savefig('test.eps')

def test_minuit():

    X = np.linspace(-5, 5, 100)

    flux = lambda x, y: np.ones_like(X)*1 + x*np.exp(-(X-y)**2/2)
    noise = flux(1., 0.)*0
    exposure = np.ones_like(X)*0.1
    M = get_minuit(flux, noise, exposure, [1., 0.], [0.01, 0.01], print_level = 1)
    M.migrad()
    M.minos()
    templates = func_to_templates(flux, [1., 0.], [0.0001, 0.0001])
    #plt.plot(templates[0])
    #plt.plot(templates[1])
    #plt.show()
    noise = flux(1.,0)
    rf = Rockfish(templates, noise, None, exposure)
    sigma1 = 1./rf.effectivefishermatrix(0)**0.5
    sigma2 = 1./rf.effectivefishermatrix(1)**0.5
    print "Minos:", M.merrors[('x1',1.0)], M.merrors[('x2',1.0)]
    print "Minos:", M.merrors[('x1',-1.0)], M.merrors[('x2',-1.0)]
    print "Hesse:", M.errors['x1'], M.errors['x2']
    print "Rockfish:", sigma1, sigma2
    print "Events:", (exposure*flux(1,0)).sum()

    #print 1/rf.fishermatrix()[0,0]**0.5, 1/rf.fishermatrix()[1,1]**0.5
    #x, y, v = M.contour("x1", "x2")
    #import pylab as plt
    #plt.contour(x, y, v)
    #plt.show()


def test_minuit_contours():

    X = np.linspace(-5, 5, 100)

    flux = lambda x, y:  np.ones_like(X)*1 + x*np.exp(-(X-y)**2/2)
    noise = flux(1., 0.)*0
    exposure = np.ones_like(X)*1.0
    M = get_minuit(flux, noise, exposure, [1., 0.], [0.01, 0.01], print_level = 1)
    M.migrad()
    M.minos()
    templates = func_to_templates(flux, [1., 0.], [0.0001, 0.0001])
    #plt.plot(templates[0])
    #plt.plot(templates[1])
    #plt.show()
    noise = flux(1.,0)
    rf = Rockfish(templates, noise, None, exposure)
    sigma1 = 1./rf.effectivefishermatrix(0)**0.5
    sigma2 = 1./rf.effectivefishermatrix(1)**0.5
#    print "Minos:", M.merrors[('x1',1.0)], M.merrors[('x2',1.0)]
#    print "Minos:", M.merrors[('x1',-1.0)], M.merrors[('x2',-1.0)]
#    print "Hesse:", M.errors['x1'], M.errors['x2']
#    print "Rockfish:", sigma1, sigma2
#    print "Events:", (exposure*flux(1,0)).sum()

    print 1/rf.fishermatrix()[0,0]**0.5, 1/rf.fishermatrix()[1,1]**0.5
    x, y, v = M.contour("x1", "x2", bound=4)
    import pylab as plt
    plt.contour(x, y, v, [1,4,9])
    plt.xlim([0, 2])
    plt.ylim([-1, 1])
    plt.savefig('test.eps')


def test_convolution():
    E = Logbins(1, 3, 100)
    K = Convolution1D(E, 0.1)
    mu = np.zeros(100)+0.1
    mu[50] = 1
    mu[30] = 1
    mu[70] = 1
    plt.semilogx(E.means, K(mu))
    print K(mu).sum()
    plt.show()
    quit()


if __name__ == "__main__":
    #test_3d()
    #test_covariance()
    #test_simple()
    #test_UL()
    #smoothtest()
    #test_spectra()
    #test_matrix()
    #test_infoflux()
    #test_minuit()
    test_minuit_contours()
    #test_convolution()
