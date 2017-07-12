#!/usr/bin/env python3

import numpy as np
import sys
import os.path

def header(datfile):

    CH = '.'.join(os.path.basename(datfile).split('.')[:-1]).upper()
    ch = '.'.join(os.path.basename(datfile).split('.')[:-1]).lower()

    d = np.loadtxt(datfile)

    m = np.unique(d[:,0])
    logx = np.unique(d[:,1])

    output = '#define MSIZE_%s %i\n#define XSIZE_%s %i\n' % (CH, m.shape[0], CH, logx.shape[0])

    output += '\nint msize_%s = MSIZE_%s;\nint xsize_%s = XSIZE_%s;\n' % (ch, CH, ch, CH)

    output += "\ndouble mass_%s[MSIZE_%s] = {\n" % (ch, CH)
    output += ''.join(['\t%.6e,\n' % v for v in m])
    output += '};\n'

    output += "\ndouble logx_%s[XSIZE_%s] = {\n" % (ch, CH)
    output += ''.join(['\t%.6e,\n' % v for v in logx])
    output += '};\n'

    output += "\ndouble dNdlogx_%s[MSIZE_%s*XSIZE_%s] = {\n" % (ch, CH, CH)
    output += ''.join(['\t%.6e,\n' % v for v in d[:,2]])
    output += '};\n'

    return output

def headers(channels):
    for ch in channels:
        o = header('../data/%s.dat' % ch)
        with open('%s.h' % ch, 'w') as fd:
            fd.write(o)

def includes(channels):
    return ''.join(['#include "%s.h"\n' % ch for ch in channels])

def declarations(channels):
    code = """
double interp_%s(double, double);
double dNdE_%s(double *, double *);
"""

    return ''.join([code % (ch, ch) for ch in channels])


def functions(channels):

    code = """
double
interp_%s(double m, double lx) {
    return interp(m, lx, mass_%s, logx_%s, dNdlogx_%s, MSIZE_%s, XSIZE_%s);
}

double
dNdE_%s(double *m, double *e) {
    return dNdE(*m, *e, &interp_%s);
}
"""
    return ''.join([code % (ch, ch, ch, ch, ch.upper(), ch.upper(), ch, ch)
                    for ch in channels])

def interp_c(channels):
    code = """
#include <math.h>

%s

#define eps 0.000001

double interp(double, double, double *, double *, double *, int, int);
double dNdE(double, double, double (*)(double, double));
%s

double
dNdE(double m, double e, double (* I)(double, double)) {
    double lx, dNdlx;

    if (e + eps > m) {
        return 0.0;
    }

    lx = log10(e/m);

    dNdlx = (*I)(m, lx);

    return dNdlx / (e * log(10));

}

double
interp(double m, double lx, double *m_t, double *lx_t, double *dNdlx_t,
        int m_size, int lx_size) {

    int mi, lxi, i;
    double mdiff, lxdiff;
    double fxy1, fxy2, fxy;

    if (m < m_t[0] || m > m_t[m_size-1]) {
        return -1;
    }

    if (lx < lx_t[0] || lx > lx_t[lx_size-1]) {
        return -1;
    }

    for (i = 0; i < m_size; i++) {
        if (m < m_t[i]+eps) {
            mi = i-1;
            break;
        }
    }

    for (i = 0; i < lx_size; i++) {
        if (lx < lx_t[i]+eps) {
            lxi = i-1;
            break;
        }
    }

    mdiff = m_t[mi+1] - m_t[mi];
    lxdiff = lx_t[lxi+1] - lx_t[lxi];

    /* First interpolate in x direction (i.e. the mass) */
    fxy1 = dNdlx_t[mi*lx_size+lxi] * (m_t[mi+1] - m)/mdiff + dNdlx_t[(mi+1)*lx_size+lxi] * (m - m_t[mi])/mdiff;

    fxy2 = dNdlx_t[mi*lx_size+lxi+1] * (m_t[mi+1] - m)/mdiff + dNdlx_t[(mi+1)*lx_size+lxi+1] * (m - m_t[mi])/mdiff;

    /* And now in the y direction (i.e. logx)*/
    fxy = fxy1 * (lx_t[lxi+1] - lx)/lxdiff + fxy2 * (lx - lx_t[lxi])/lxdiff;

    return fxy;
}
%s
""" % (includes(channels), declarations(channels), functions(channels))

    with open('interp.c', 'w') as fd:
        fd.write(code)


def main():
    channels = ['tt', 'bb', 'cc', 'qq', 'gg']
    interp_c(channels)
    headers(channels)

    return 0

if __name__ == '__main__':
    main()
