import numpy as np
from scipy import interpolate
import os.path

class Interp:
    """
    A Interp object is a interpolator for the production fluxes for a specific
    annhilation channel. Given dark matter masses and photon energies it will
    give you the fluxes. The tables being interpolated is from PPPC 4 DM ID.

    The interpolation algorithm used is bilinear. The main work is done in C.
    """

    def __init__(self, ch):
        from ctypes import cdll, c_double, c_int, POINTER

        lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'cinterp/libpppc.so'))

        msize = c_int.in_dll(lib, 'msize_%s' % ch).value
        xsize = c_int.in_dll(lib, 'xsize_%s' % ch).value

        self.m = np.array((c_double*msize).in_dll(lib, 'mass_%s' % ch))
        self.log10x = np.array((c_double*xsize).in_dll(lib, 'logx_%s' % ch))

        if ch == 'tt':
            self.cdNdE = lib.dNdE_tt
            self.cinterp = lib.interp_tt
        if ch == 'bb':
            self.cdNdE = lib.dNdE_bb
            self.cinterp = lib.interp_bb
        if ch == 'cc':
            self.cdNdE = lib.dNdE_cc
            self.cinterp = lib.interp_cc
        if ch == 'qq':
            self.cdNdE = lib.dNdE_qq
            self.cinterp = lib.interp_qq
        if ch == 'gg':
            self.cdNdE = lib.dNdE_gg
            self.cinterp = lib.interp_gg

        self.cdNdE.restype = c_double
        self.cinterp.restype = c_double

        #c_double_p = lambda x : POINTER(c_double(x))
        c_double_p = POINTER(c_double)
        self.interp = np.vectorize(lambda m, lx: self.cinterp(c_double(m), c_double(lx)))
        self.dNdE = np.vectorize(lambda m, e: self.cdNdE(c_double_p(c_double(m)), c_double_p(c_double(e))))


    def __call__(self, masses, energies):
        """ Returns dNdE for given arrays of masses and energies.
        """
        return self.dNdE(masses, energies)


def defaultdatpath(ch):
    return os.path.join(os.path.dirname(__file__), 'data/%s.dat' % ch)


# Supported channels.
channels = ['gg', 'tt', 'bb', 'cc', 'qq']

def loadInterpolators(channels=channels):
    d = {}
    for ch in channels:
        d[ch] = Interp(ch)

    return d


###
# Below follows various plotting functions which are useful for debugging,
# and sanity checks.
###

def plot_flux(mass, e=None, emin = 0.1, emax=1000, enum=1000, channels = channels, fig_path=None):
    import matplotlib.pyplot as plt

    if e == None:
        e = np.linspace(emin, emax, enum)

    spectra = loadInterpolators()

    fig = plt.figure()

    for ch in channels:
        plt.plot(e, e * e * spectra[ch](mass, e), label=ch)

    plt.legend()
    plt.xlabel('E [GeV]')
    plt.ylabel('E * E * dNdE')

    plt.xscale('log')

    try:
        plt.yscale('log')
    except ValueError: # all y values are zero.
        plt.yscale('linear')

    if fig_path == None:
        fig_path = '%.1f-fluxes.png' % mass

    fig.savefig(fig_path)

    plt.close(fig)


def plot_interp(ch, fig_path=None, N=1000, xlim=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    I = Interp(ch)

    m = np.linspace(I.m.min(), I.m.max(), N)
    log10x = np.linspace(I.log10x.min(), I.log10x.max(), N)

    M = np.repeat(m[:,np.newaxis], N, axis=1)
    LOG10X = np.repeat(log10x[:,np.newaxis], N, axis=1).T

    dNdlog10x = I.interp(M, LOG10X)

    fig = plt.figure()

    plt.pcolormesh(dNdlog10x.T, norm=LogNorm(vmin=1e-6, vmax=dNdlog10x.max()), cmap='PuBu_r')

    plt.colorbar()

    datfile = os.path.join(os.path.dirname(__file__), 'data/%s.dat' % ch)
    m, log10x, n = np.loadtxt(datfile).T

    m = np.unique(m)
    log10x = np.unique(log10x)

    plt.xticks(N/I.m.max()*I.m, I.m)
    plt.yticks(N - N/I.log10x.min()*I.log10x, I.log10x)
    plt.grid()

    if xlim != None:
        plt.xlim(xlim[0], xlim[1])

    fig.set_size_inches(60,45)

    if fig_path == None:
        fig_path = '%s-interp.png' % ch

    fig.savefig(fig_path)

    plt.close(fig)


def plot_dat(ch, dat_path=None, fig_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if dat_path == None:
        dat_path = defaultdatpath(ch)

    m, log10x, n = np.loadtxt(dat_path).T

    cols = np.unique(log10x).shape[0]

    M = m.reshape(-1,cols)
    LOG10X = log10x.reshape(-1,cols)
    N = n.reshape(-1,cols)

    fig = plt.figure()

    plt.pcolormesh(N.T, norm=LogNorm(vmin=1e-10, vmax=N.max()), cmap='PuBu_r')
    plt.colorbar()

    if fig_path == None:
        fig_path = '%s-dat.png' % ch

    fig.savefig(fig_path)

    plt.close(fig)


def plot_interp_dat_diff(ch, dat_path=None, fig_path=None):
    import matplotlib.pyplot as plt

    if dat_path == None:
        dat_path = defaultdatpath(ch)

    m, log10x, n = np.loadtxt(dat_path).T

    m = np.unique(m)
    log10x = np.unique(log10x)

    cols = log10x.shape[0]
    N = n.reshape(-1,cols).T

    M = np.repeat(m[:,np.newaxis], log10x.shape[0], axis=1)
    LOG10X = np.repeat(log10x[:,np.newaxis], m.shape[0], axis=1).T

    I = Interp(ch)

    Ninterp = I.interp(M, LOG10X)

    Ndiff = N - Ninterp.T

    fig = plt.figure()

    plt.pcolormesh(Ndiff, cmap='PuBu_r')
    plt.colorbar()

    if fig_path == None:
        fig_path = '%s-check.png' % ch

    fig.savefig(fig_path)

    plt.close(fig)


def plot_everything():
    for ch in channels:
        plot_interp(ch)
        plot_dat(ch)
        plot_interp_dat_diff(ch)
