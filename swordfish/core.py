#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
from scipy import stats
import copy
from scipy.special import gammaln
from scipy.optimize import fmin_l_bfgs_b
import metricplot as mp

def _init_minuit(f, x = None, x_fix = None, x_err = None, x_lim = None, errordef = 1, **kwargs):
    """Initialize minuit using no-nonsense interface."""
    try:
        import iminuit
    except ImportError:
        raise ImportError(
                "This function requires that the module iminuit is installed.")

    N = len(x)
    if x_err is not None:
        assert len(x_err) == N
    if x_lim is not None:
        assert len(x_lim) == N
    if x_fix is not None:
        assert len(x_fix) == N
    varnames = ["x"+str(i) for i in range(N)]
    def wf(*args):
        x = np.array(args)
        return f(x)
    for i, var in enumerate(varnames):
        kwargs[var] = x[i]
        if x_lim is not None:
            kwargs["limit_"+var] = x_lim[i]
        if x_err is not None:
            kwargs["error_"+var] = x_err[i]
        if x_fix is not None:
            kwargs["fix_"+var] = x_fix[i]
    return iminuit.Minuit(wf, forced_parameters = varnames, errordef =
            errordef, **kwargs)

def get_minuit(flux, noise, exposure, thetas0, thetas0_err, **kwargs):
    """Create iminuit.Minuit object from input data.

    The intended use of this function is to allow an easy cross-check of the
    SwordFish results (in absence of systematic uncertainties).

    Arguments
    ---------
    flux : function of model parameters, returns 1-D array
        Flux.
    noise : 1-D array
        Noise.
    exposure : {float, 1-D array}
        Exposure.
    thetas0: 1-D array
        Initial values, define mock data.
    thetas0_err: 1-D array
        Initial errors.
    **kwargs: additional arguments
        Passed on to iminuit.Minuit.

    Note
    ----
        The (expected) counts in the Poisson likelikhood are given by

            mu(thetas) = exposure*(flux(thetas)+noise) .
    """
    def chi2(thetas):
        mu = (flux(*thetas) + noise)*exposure
        mu0 = (flux(*thetas0) + noise)*exposure
        lnL = -(mu*np.log(mu/mu0)-mu+mu0)
        return -lnL.sum()*2

    M = _init_minuit(chi2, x = thetas0, x_err = thetas0_err, **kwargs)
    return M

def func_to_templates(flux, x, dx = None):
    """Return finite differences for use in SwordFish."""
    x = np.array(x, dtype='float64')
    if dx is None:
        dx = x*0.01+0.001
    fluxes = []
    #fluxes.append(flux(*x))
    for i in range(len(x)):
        xU = copy.copy(x)
        xL = copy.copy(x)
        xU[i] += dx[i]
        xL[i] -= dx[i]
        df = (flux(*xU) - flux(*xL))/2./dx[i]
        fluxes.append(df)
    return fluxes

class FunkFish(object):
    """Docstring for FunkFish"""
    def __init__(self, f, Sigma, exposure, x0, constraints = None):
        self._f = f
        self._Sigma = Sigma
        self._exposure = exposure
        self._x0 = np.array(x0, dtype='float64')
        self._constraints = constraints

    def _get_x0(self, x0_dict = {}):
        """Get updated x0."""
        x0 = self._x0.copy()
        for i in x0_dict:
            x0[i] = x0_dict[i]
        return x0

    def get_Swordfish(self, x0_dict = {}):
        x0 = self._get_x0(x0_dict)
        flux = func_to_templates(self._f, x0)
        noise = self._f(*x0)
        return Swordfish(flux, noise, self._Sigma, self._exposure, constraints
                = self._constraints)

    def get_EffectiveCounts(self, x0_dict = {}):
        SF = self.get_Swordfish(x0_dict)
        return EffectiveCounts(SF)

    def get_TensorFields(self, i_x, i_y, x_bins, y_bins, x0_dict = {}):
        x0_dict = x0_dict.copy()
        g = np.zeros((len(y_bins), len(x_bins), 2, 2))
        for i, y in enumerate(y_bins):
            for j, x in enumerate(x_bins):
                x0_dict[i_x] = x
                x0_dict[i_y] = y
                SF = self.get_Swordfish(x0_dict)
                g[i, j] = SF.effectivefishermatrix((i_x, i_y))
        return mp.TensorField(x_bins, y_bins, g)

    def get_iminuit(self, x0_dict = {}):
        SF = self.get_Swordfish(x0_dict)
        x0 = self._get_x0(x0_dict)
        def chi2(x):
            lnL = SF.profile_lnL(x-x0, x0*0.)
            return -2*lnL
        x0err = np.where(x0>0., x0*0.01, 0.01)
        M = _init_minuit(chi2, x = x0, x_err = x0err)
        return M

class Swordfish(object):  # Everything is flux!
    """Swordfish(flux, noise, systematics, exposure, solver = 'direct', verbose = False)
    """
    def __init__(self, flux, noise, systematics, exposure, constraints = None, solver = 'direct',
            verbose = False):
        """Construct swordfish model from input.

        Arguments
        ---------
        flux : list of 1-D arrays
        noise : 1-D array
        systematics : {sparse matrix, dense matrix, LinearOperator, None}
        exposure : {float, 1-D array}
        solver : {'direct', 'cg'}, optional
        verbose : bool, optional
        """
        self._flux = flux
        self._noise = noise
        self._exposure = exposure
        self._cache = None
        self._verbose = verbose
        self._scale = self._get_auto_scale(flux, exposure)
        self._solver = solver
        self._nbins = len(self._noise)  # Number of bins
        self._ncomp = len(self._flux)   # Number of flux components
        self._sysflag = systematics is not None
        if systematics is not None:
            self._systematics = la.aslinearoperator(systematics)
        else:
            self._systematics = la.LinearOperator((self._nbins, self._nbins), 
                    matvec = lambda x: np.zeros_like(x))
        self._constraints = self._get_constraints(constraints, self._ncomp)

    @staticmethod
    def _get_auto_scale(flux, exposure):
        return np.array(
                [1./(f*exposure).max() for f in flux]
                )

    @staticmethod
    def _get_constraints(constraints, ncomp):
        assert constraints is None or len(constraints) == ncomp
        if constraints is not None:
            constraints = np.array(
                [np.inf if x is None or x == np.inf else x for x in constraints]
                )
            if any(constraints<=0.):
                raise ValueError("Constraints must be positive or None.")
        else:
            constraints = np.ones(ncomp)*np.inf
        return constraints

    def _summedNoise(self, thetas = None):
        noise_tot = self._noise*1.  # Make copy
        if thetas is not None: 
            for i in range(max(self._ncomp, len(thetas))):
                noise_tot += thetas[i]*self._flux[i]
        return noise_tot

    def _solveD(self, thetas = None):
        """
        Calculates:
            N = noise + thetas*flux
            D = diag(E)*Sigma*diag(E)+diag(N*E)
            x[i] = D^-1 flux[i]*E

        Note: if Sigma = None: x[i] = flux[i]/noise

        Returns:
            x, noise, exposure
        """
        noise = self._summedNoise(thetas)
        exposure = self._exposure
        spexp = la.aslinearoperator(sp.diags(exposure))
        D = (
                la.aslinearoperator(sp.diags(noise*exposure))
                + spexp*self._systematics*spexp
                )
        x = np.zeros((self._ncomp, self._nbins))
        if self._solver == "direct":
            dense = D(np.eye(self._nbins))
            invD = np.linalg.linalg.inv(dense)
            for i in range(self._ncomp):
                x[i] = np.dot(invD, self._flux[i]*exposure)*exposure
        elif self._solver == "cg":
            def callback(x):
                if self._verbose:
                    print len(x), sum(x), np.mean(x)
            for i in range(self._ncomp):
                x0 = self._flux[i]/noise if self._cache is None else self._cache/exposure
                x[i] = la.cg(D, self._flux[i]*exposure, x0 = x0, callback = callback, tol = 1e-3)[0]
                x[i] *= exposure
                self._cache= x[i]
        else:
            raise KeyError("Solver unknown.")
        return x, noise, exposure

    def fishermatrix(self, thetas = None):
        """Return Fisher Information Matrix.

        Arguments
        ---------
        thetas : array-like, optional
            Flux components added to noise during evaluation.
        """
        x, noise, exposure = self._solveD(thetas=thetas)
        I = np.zeros((self._ncomp,self._ncomp))
        for i in range(self._ncomp):
            for j in range(i+1):
                tmp = sum(self._flux[i]*x[j])
                I[i,j] = tmp
                I[j,i] = tmp
        return I+np.diag(1./self._constraints**2)

    def infoflux(self, thetas = None):
        """Return Fisher Information Flux.

        Arguments
        ---------
        thetas : array-like, optional
            Flux components added to noise during evaluation.
        """
        x, noise, exposure = self._solveD(thetas=thetas)

        F = np.zeros((self._ncomp,self._ncomp,self._nbins))
        for i in range(self._ncomp):
            for j in range(i+1):
                tmp = x[i]*x[j]*noise/(exposure**2.)
                F[i,j] = tmp
                F[j,i] = tmp
        return F

    def effectivefishermatrix(self, indexlist, **kwargs):
        """Return variance of theta.

        Arguments
        ---------
        i : integer
            index of component of interest, or list of indices
        **kwargs
            Passed on to fishermatrix
        """
        if isinstance(indexlist, np.int):
            indexlist = [indexlist]
        indices = np.setdiff1d(range(self._ncomp), indexlist)
        n = len(indexlist)

        I = self.fishermatrix(**kwargs)
        A = I[indexlist,:][:,indexlist]
        B = I[indices,:][:,indexlist]
        C = I[indices,:][:,indices]
        invC = np.linalg.linalg.inv(C)

        Ieff = A - B.T.dot(invC.dot(B))

        return Ieff

    def variance(self, i, **kwargs):
        """Return variance of theta.

        Arguments
        ---------
        i : integer
            index of component of interest
        **kwargs
            Passed on to fishermatrix
        """
        I = self.fishermatrix(**kwargs)
        invI = np.linalg.linalg.inv(I)
        return invI[i,i]

    def effectiveinfoflux(self, i, **kwargs):
        """Return effective Fisher Information Flux.

        Arguments
        ---------
        i : integer
            index of component of interest
        **kwargs
            Passed on to fishermatrix
        """
        F = self.infoflux(**kwargs)
        I = self.fishermatrix(**kwargs)
        n = self._ncomp
        if n == 1:
            return F[i,i]
        indices = np.setdiff1d(range(n), i)
        eff_F = F[i,i]
        C = np.zeros(n-1)
        B = np.zeros((n-1,n-1))
        for j in range(n-1):
            C[j] = I[indices[j],i]
            for k in range(n-1):
                B[j,k] = I[indices[j], indices[k]]
        invB = np.linalg.linalg.inv(B)
        for j in range(n-1):
            for k in range(n-1):
                eff_F = eff_F - F[i,indices[j]]*invB[j,k]*C[k]
                eff_F = eff_F - C[j]*invB[j,k]*F[indices[k],i]
                for l in range(n-1):
                    for m in range(n-1):
                        eff_F = eff_F + C[j]*invB[j,l]*F[indices[l],indices[m]]*invB[m,k]*C[k]
        return eff_F

    # Likelihood with systematics
    def lnL(self, thetas, thetas0, dmu = None, epsilon = 1e-3, derivative = False):
        """Return likelihood function, assuming Asimov data.

        Arguments
        ---------
        thetas : array-like, optional
            Definition of flux (added to noise during evaluation).
        thetas0 : array-like, optional
            Definition of Asimov data.
        dmu : array-like
            Systematic deviation.
        epsilon : Float
            Fraction of diagonal noise added to Sigma for stable matrix
            inversion.  Default is 1e-2.
        derivative: bool
            Return also partial derivative w.r.t. thetas and w.r.t. dmu
        """
        mu0 = self._summedNoise(thetas0)*self._exposure
        systnoise = self._summedNoise(thetas0)*epsilon/self._exposure
        mu =  self._summedNoise(thetas)*self._exposure
        if dmu is None:
            dmu = np.zeros_like(self._exposure)
        if self._sysflag:
            mu += dmu*self._exposure
        self._at_bound = any(mu<mu0*1e-6)
        mu = np.where(mu<mu0*1e-6, mu0*1e-6, mu)
        lnL = (mu0*np.log(mu)-mu-gammaln(mu0+1)).sum()
        lnL -= (0.5*thetas**2/self._constraints**2).sum()
        if self._sysflag:
            dense = self._systematics(np.eye(self._nbins))
            #invS = np.linalg.linalg.inv(dense+np.eye(self._nbins)*epsilon)
            invS = np.linalg.linalg.inv(dense+np.diag(systnoise))
            lnL -= 0.5*(invS.dot(dmu)*dmu).sum()
        if derivative:
            dlnL_dtheta = (mu0/mu*self._flux*self._exposure-self._flux*self._exposure).sum(axis=1)
            dlnL_dtheta -= thetas/self._constraints**2
            if self._sysflag:
                dlnL_dmu = mu0/mu*self._exposure - self._exposure - invS.dot(dmu)
            else:
                dlnL_dmu = None
            return lnL, dlnL_dtheta, dlnL_dmu
        else:
            return lnL

    def profile_lnL(self, thetas, thetas0, epsilon = 1e-3, free_thetas = None):
        """Return profile likelihood.

        Arguments
        ---------
        free_thetas : array-like, boolean
            Parameters kept fixed during maximization
        """
        thetas = np.array(thetas, dtype='float64')
        thetas0 = np.array(thetas0, dtype='float64')
        if free_thetas is None:
            free_thetas = np.zeros(len(thetas), dtype='bool')
        else:
            free_thetas = np.array(free_thetas, dtype='bool')
        Nfree_thetas = (free_thetas).sum()
        Nsyst = len(self._exposure) if self._sysflag else 0
        thetas = thetas.copy()
        N = Nfree_thetas + Nsyst
        x0 = np.zeros(N)
        x0[:Nfree_thetas] = thetas[free_thetas]
        fp = 0.
        def f(x):
            global fp
            thetas[free_thetas] = x[:Nfree_thetas]
            dmu = x[Nfree_thetas:]
            lnL, grad_theta, grad_dmu = self.lnL(thetas, thetas0, dmu = dmu,
                    epsilon = epsilon, derivative = True)
            fp = np.zeros(N)
            fp[:Nfree_thetas] = grad_theta[free_thetas]
            fp[Nfree_thetas:] = grad_dmu
            return -lnL
        def fprime(x):
            global fp
            return -fp
        if N == 0.:
            return self.lnL(thetas, thetas0, dmu = None)
        result = fmin_l_bfgs_b(f, x0, fprime, approx_grad = False)
        if self._verbose:
            print "Best-fit parameters:", result[0]
        if self._at_bound:
            print "WARNING: No maximum with non-negative flux found."
            return None
        else:
            return -result[1]

class EffectiveCounts(object):
    """EffectiveCounts(model).
    """
    def __init__(self, model):
        """Construct EffectiveCounts object.

        Paramters
        ---------
        model : Swordfish
            Input Swordfish model.

        Note: The functionality applies *only* to additive component models.
        You have been warned.
        """
        self._model = model

    def noise_counts(self):
        """Return total counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        lambda : float
            Number of counts in component i.
        """
        return sum(self._model._noise*self._model._exposure)

    def counts(self, i, theta):
        """Return total counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        lambda : float
            Number of counts in component i.
        """
        return sum(self._model._flux[i]*self._model._exposure*theta)

    def effectivecounts(self, i, theta):
        """Return effective counts.

        Parameters
        ----------
        i : integer
            Component of interest.
        theta : float
            Normalization of component i.

        Returns
        -------
        s : float
            Effective signal counts.
        b : float
            Effective backgroundc counts.
        """
        I0 = 1./self._model.variance(i)
        thetas = np.zeros(self._model._ncomp)
        thetas[i] = theta
        I = 1./self._model.variance(i, thetas = thetas)
        if I0 == I:
            return 0., None
        s = 1/(1/I-1/I0)*theta**2
        b = 1/I0/(1/I-1/I0)**2*theta**2
        return s, b

    def upperlimit(self, alpha, i, gaussian = False):
        """Returns upper limits, based on effective counts method.

        Parameters
        ----------
        alpha : float
            Statistical significance (e.g., 95% CL is 0.05).
        i : integer
            Component of interest.
        gaussian : bool, optional
            Force gaussian errors.

        Returns
        -------
        thetaUL : float
            Predicted upper limit on component i.
        """
        Z = stats.norm.isf(alpha)
        I0 = 1./self._model.variance(i)
        if gaussian:
            return Z/np.sqrt(I0)
        else:
            thetas = np.zeros(self._model._ncomp)
            thetaUL_est = Z/np.sqrt(I0)  # Gaussian estimate
            thetas[i] = thetaUL_est
            I = 1./self._model.variance(i, thetas = thetas)
            if (I0-I)<0.02*I:  # 1% accuracy of limits
                thetaUL = thetaUL_est
            else:
                z_list = []
                theta_list = [thetaUL_est]
                while True:
                    theta = theta_list[-1]
                    s, b = self.effectivecounts(i, theta = theta)
                    if s == 0: b = 1.
                    z_list.append(s/np.sqrt(s+b))
                    if z_list[-1] > Z:
                        break
                    else:
                        theta_list.append(theta*1.3)
                    #print theta, z_list
                thetaUL = np.interp(Z, z_list, theta_list)
            return thetaUL

    def discoveryreach(self, alpha, i, gaussian = False):
        Z = stats.norm.isf(alpha)
        var0 = self._model.variance(i)
        if gaussian:  # Gaussian approximation
            return Z*var0**0.5

        thetas = np.zeros(self._model._ncomp)
        thetaDT_est = Z*np.sqrt(var0)  # Gaussian approx. as starting point
        thetas[i] = thetaDT_est
        var = self._model.variance(i, thetas = thetas)
        if abs(var0 - var) < 0.02*var:  # Still Gaussian enough
            return Z*var0**0.5
        z_list = []
        theta_list = [thetaDT_est]
        while True:
            theta = theta_list[-1]
            s, b = self.effectivecounts(i, theta = theta)
            if s == 0: b = 1.
            z_list.append((s+b)*np.log((s+b)/b)-s)
            if z_list[-1] > Z**2/2:
                break
            else:
                theta_list.append(theta*1.3)
        return np.interp(Z**2/2, z_list, theta_list)

### Obsolete
#
#class Visualization(object):
#    def __init__(self, xy, I11, I22, I12):
#        pass
#
#    def plot(self):
#        pass
#
#    def integrate(self):
#        pass
#
#def tensorproduct(Sigma1, Sigma2):
#    Sigma1 = la.aslinearoperator(Sigma1)
#    Sigma2 = la.aslinearoperator(Sigma2)
#    n1 = np.shape(Sigma1)[0]
#    n2 = np.shape(Sigma2)[0]
#    Sigma2 = Sigma2(np.eye(n2))
#    N = n1*n2
#    def Sigma(x):
#        A = np.reshape(x, (n1, n2))
#        B = np.zeros_like(A)
#        for i in range(n2):
#            y = Sigma1(A[:,i])
#            for j in range(n2):
#                B[:,j] += Sigma2[i,j]*y
#        return np.reshape(B, N)
#    return la.LinearOperator((N, N), matvec = lambda x: Sigma(x))
