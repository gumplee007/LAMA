# This code is used to normalize a spectrum with local regresssion fitting method
# import module
import numpy as np
from scipy.interpolate import interp1d
from localreg import *

def norm(x, y, k = 2/5, upper = 7, lower = 0.5, iter_num = 3):
    '''
    input:
    x, y: the wavelength and flux of a spectrum
    k: the partiton of the range of wavelength, it is used to set the parameter m0 in function localreg, the default value of k is 2/5.
    upper, lower: the upper and lower limit to exlude the emission and absorption lines, default is 7 and 0.5 times of the standard deviation of the reidual.
    iter_num: iteration times, the default is 3.
    output:
    B_flux: the continuum
    x: the wavelength of the normalized spectrum
    y/B_flux: flux of the normalized spectrum
    index_cr: the index which indicate the spectral array without cosmic-ray
    index_em: the index which indicate the spectral array without emission lines
    '''
    # set m0
    m0 = k*(np.max(x) - np.min(x))
    # a 1D array of wavelength including 100 points, the purpose is reducing computing time
    wl_dummy = np.linspace(np.min(x), np.max(x), 100)
    # remove the Ha line between 6520 and 6610 angstorm.
    index = (x < 6520) | (x > 6610)
    xx = x[index]
    yy = y[index]
    # iterate iter_num times:
    for i in range(iter_num):
        # use local regression function localreg to fit the spectral continuum, the output continuum is corresponding to the wavelength wl_dummy
        B_flux = localreg(xx, yy, x0 = wl_dummy, degree = 3, kernel = rbf.tricube, radius = m0)
        # use a the quadratic spline function to upsample the continuum from 100 points to the original spectrum
        func_spline = interp1d(wl_dummy, B_flux, kind = 'quadratic')
        B_flux = func_spline(xx)
        # the standard deviation of the residual between the spectrum and the continuum
        e = np.std(yy - B_flux)
        # remove the emission and absorption lines
        index = (yy < B_flux + upper*e) & (yy > B_flux - lower*e)
        xx = xx[index]
        yy = yy[index]
    # the final continuum
    B_flux = func_spline(x)
    # find the cosmic-ray which the emission line has less than 10 continuous points
    index_cr = np.ones(len(x), dtype = bool)
    index = np.append(np.where(y > B_flux + upper*e)[0], [99999])
    idx0 = 0
    for i in range(len(index) - 1):
        if((index[i + 1] - index[i]) > 1.5):
            if(i - idx0 <= 10):
                index_cr[index[idx0:i + 1]] = False
            idx0 = i + 1
    # the other emission lines excluding the cosmic-rays
    index_em = y[index_cr] < B_flux[index_cr] + upper*e
    return B_flux, x, y/B_flux, index_cr, index_em
