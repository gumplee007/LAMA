import numpy as np
import pandas as pd
from astropy.stats import biweight_location as bl
from scipy.interpolate import interp1d
from localreg import *

def norm(x, y, k = 2/5, upper = 7, lower = 0.5, iter_num = 3):
    m0 = k*(np.max(x) - np.min(x))
    wl_dummy = np.linspace(np.min(x), np.max(x), 100)

    index = (x < 6520) | (x > 6610)
    xx = x[index]
    yy = y[index]

    for i in range(iter_num):
        B_flux = localreg(xx, yy, x0 = wl_dummy, degree = 3, kernel = rbf.tricube, radius = m0)
        func_spline = interp1d(wl_dummy, B_flux, kind = 'quadratic')
        B_flux = func_spline(xx)
        e = np.std(yy - B_flux)
        index = (yy < B_flux + upper*e) & (yy > B_flux - lower*e)
        xx = xx[index]
        yy = yy[index]

    B_flux = func_spline(x)

    index_cr = np.ones(len(x), dtype = bool)
    index = np.append(np.where(y > B_flux + upper*e)[0], [99999])
    idx0 = 0
    for i in range(len(index) - 1):
        if((index[i + 1] - index[i]) > 1.5):
            if(i - idx0 <= 10):
                index_cr[index[idx0:i + 1]] = False
            idx0 = i + 1

    index_em = y[index_cr] < B_flux[index_cr] + upper*e
    return B_flux, x, y/B_flux, index_cr, index_em
