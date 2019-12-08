import numpy as np
import scipy.stats

import PyEMD
import pywt
import neurokit
import entropy

import logging

import statsmodels.api

__all__= ['Feature_FractalDimensionHiguchi', 'Feature_InterquartileRange', 'Feature_MeanAbsoluteDeviation', 'Feature_MeanEnergy',
            'Feature_TeagerKaiserEnergy', 'Feature_SampleEntropy', 'Feature_StandardDeviation', 'Feature_DFA', 'Feature_AR_Yule_Walker']

def gen_nleo(x, l=1, p=2, q=0, s=3):
    N = len(x)
    x_nleo = np.zeros(N)

    iedges = abs(l) + abs(p) + abs(q) + abs(s)
    n = np.arange(iedges + 1, (N - iedges - 1))

    x_nleo[n] = x[n-l] * x[n-p] - x[n-q] * x[n-s]

    return(x_nleo)


def specific_nleo(x, type='teager'):
    """ generate different NLEOs based on the same operator 
    Parameters
    ----------
    x: ndarray
        input signal
    type: {'teager', 'agarwal', 'palmu', 'abs_teager', 'env_only'}
        which type of NLEO? 
    Returns
    -------
    x_nleo : ndarray
        NLEO array
    """

    def teager():
        return(gen_nleo(x, 0, 0, 1, -1))

    def agarwal():
        return(gen_nleo(x, 1, 2, 0, 3))

    def palmu():
        return(abs(gen_nleo(x, 1, 2, 0, 3)))

    def abs_teager():
        return(abs(gen_nleo(x, 0, 0, 1, -1)))

    def env_only():
        return(abs(x) ** 2)

    def default_nleo():
        # -------------------------------------------------------------------
        # default option
        # -------------------------------------------------------------------
        print('Invalid NLEO name; defaulting to Teager')
        return(teager())

    # pick which function to execute
    which_nleo = {'teager': teager, 'agarwal': agarwal,
                  'palmu': palmu, 'abs_teager': abs_teager,
                  'env_only': env_only}

    def get_nleo(name):
        return which_nleo.get(name, default_nleo)()

    x_nleo = get_nleo(type)
    return(x_nleo)

class Feature_FractalDimensionHiguchi:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: neurokit.complexity(signal, 
                        shannon=False, 
                        sampen=False, 
                        multiscale=False, 
                        spectral=False, 
                        svd=False, 
                        correlation=False, 
                        higushi=True, 
                        petrosian=False, 
                        fisher=False, 
                        hurst=False, 
                        dfa=False, 
                        lyap_r=False, 
                        lyap_e=False, 
                        emb_dim=2)['Fractal_Dimension_Higushi']}

class Feature_InterquartileRange:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: scipy.stats.iqr(signal)}

class Feature_MeanAbsoluteDeviation:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: np.mean(np.abs(signal - np.mean(signal)))}

class Feature_MeanEnergy:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: np.mean(signal*signal)}

class Feature_TeagerKaiserEnergy:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        nleo= specific_nleo(signal)
        result= np.mean(nleo*nleo)
        return {self.__class__.__name__: result}

class Feature_SampleEntropy:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: entropy.sample_entropy(signal, order=2, metric="chebyshev")}

class Feature_StandardDeviation:
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        return {self.__class__.__name__: np.std(signal)}

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

# detrended fluctuation analysis

def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.
    
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,9], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.
    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free 
    view on neuronal oscillations, (2012).
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2 
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ] 
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))
    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show:
        fluctfit = 2**np.polyval(coeff,np.log2(scales))
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        plt.show()
    return scales, fluct, coeff[0]

class Feature_DFA:
    def __init__(self, return_fluctuations= False):
        self.return_fluctuations= return_fluctuations

    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)

        if not self.return_fluctuations:
            return {self.__class__.__name__: dfa(signal)[2]}
        else:
            _, flucts, d= dfa(signal)
            results= {self.__class__.__name__: d}
            for i, f in enumerate(flucts):
                results[self.__class__.__name__ + '_' + 'fluctuation_' + str(i)]= f
            return results

class Feature_AR_Yule_Walker:
    def __init__(self, n_features=10):
        self.n_features= n_features
    
    def extract(self, signal):
        logging.info("extracting %s" % self.__class__.__name__)
        rho, _= statsmodels.api.regression.yule_walker(signal, order=self.n_features, method="mle")
        results= {}
        for i, r in enumerate(rho):
            results[self.__class__.__name__ + '_' + str(i)]= r
        
        return results
