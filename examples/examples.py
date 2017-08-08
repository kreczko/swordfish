#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import HARPix as harp
import pylab as plt
from core import *
from tools import *


def test_spectra():
    E = Logbins(0, 3, 200)
    def flux(n1):
        return E.integrate(lambda x: 
                n1*(np.exp(-x/20)*x**-1 + 0.00003*np.exp(-(x-200)**2/2/10**2))
                    +x**-2.3*10
                    )
    templates = func_to_templates(flux, [.00001])
    noise = flux(0.)

    X, Y = np.meshgrid(np.log(E.means),np.log(E.means))
    systematics = 0.01*(
            np.diag(noise).dot(
                np.exp(-(X-Y)**2/2/0.4**2)
                )).dot(np.diag(noise))

    plt.loglog(E.means, templates[0]**2/noise*E.means**2/E.widths, label='S/N',
            lw=5)

    #systematics = None
    for expo in np.logspace(0, 4, 5):
        exposure = np.ones_like(noise)*expo
        m = Rockfish(templates, noise, systematics, exposure, solver='direct')
        f = m.effectiveinfoflux(0)
        plt.loglog(E.means, f/E.widths*E.means**2, label=expo)
    plt.xlabel("Energy")
    plt.ylabel("Information flux")

    plt.ylim([1e-4, 1e2])

    plt.legend()
    plt.savefig('test.eps')

if __name__ == "__main__":
    test_spectra()
