
#include <math.h>

#include "tt.h"
#include "bb.h"
#include "cc.h"
#include "qq.h"
#include "gg.h"


#define eps 0.000001

double interp(double, double, double *, double *, double *, int, int);
double dNdE(double, double, double (*)(double, double));

double interp_tt(double, double);
double dNdE_tt(double *, double *);

double interp_bb(double, double);
double dNdE_bb(double *, double *);

double interp_cc(double, double);
double dNdE_cc(double *, double *);

double interp_qq(double, double);
double dNdE_qq(double *, double *);

double interp_gg(double, double);
double dNdE_gg(double *, double *);


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

double
interp_tt(double m, double lx) {
    return interp(m, lx, mass_tt, logx_tt, dNdlogx_tt, MSIZE_TT, XSIZE_TT);
}

double
dNdE_tt(double *m, double *e) {
    return dNdE(*m, *e, &interp_tt);
}

double
interp_bb(double m, double lx) {
    return interp(m, lx, mass_bb, logx_bb, dNdlogx_bb, MSIZE_BB, XSIZE_BB);
}

double
dNdE_bb(double *m, double *e) {
    return dNdE(*m, *e, &interp_bb);
}

double
interp_cc(double m, double lx) {
    return interp(m, lx, mass_cc, logx_cc, dNdlogx_cc, MSIZE_CC, XSIZE_CC);
}

double
dNdE_cc(double *m, double *e) {
    return dNdE(*m, *e, &interp_cc);
}

double
interp_qq(double m, double lx) {
    return interp(m, lx, mass_qq, logx_qq, dNdlogx_qq, MSIZE_QQ, XSIZE_QQ);
}

double
dNdE_qq(double *m, double *e) {
    return dNdE(*m, *e, &interp_qq);
}

double
interp_gg(double m, double lx) {
    return interp(m, lx, mass_gg, logx_gg, dNdlogx_gg, MSIZE_GG, XSIZE_GG);
}

double
dNdE_gg(double *m, double *e) {
    return dNdE(*m, *e, &interp_gg);
}

