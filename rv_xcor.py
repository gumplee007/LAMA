# This code is use to calculate the Cross-Correlation Function between spectrum and template
import numpy as np
import scipy.constants

c = scipy.constants.speed_of_light
c = c/1000 # set 'c' is the speed of light (km/s)

def xcor(obs = None, temp = None, v = None):
    """
    Cross correlate between tempplate and observed spectrum;
    obs: (x, y)
    temp: (X, Y)
    v: (v1, v2, dv)
    return the normalized correlations of each radial velocities.
    """

    # read wavelengthes and fluxes of template and observed spectrum
    wl_obs = obs[0]
    flux_obs = obs[1]
    wl_temp = temp[0]
    flux_temp = temp[1]
    # read the edges and space of velocities
    v1 = v[0]
    v2 = v[1]
    dv = v[2]

    # if the wavelength range of template can NOT cover observed one, raise an error
    if(np.max(wl_temp) <= np.max(wl_obs)/(1 + v1/c)) | (np.min(wl_temp) >= np.min(wl_obs)/(1 + v2/c)):
        index = (wl_obs < np.max(wl_temp)/(1 + v1/c)) & (wl_obs > np.min(wl_temp)/(1 + v2/c))
        wl_obs = wl_obs[index]
        flux_obs = flux_obs[index]

    # set log wavelength all we need
    W_l = min(min(wl_obs)/(1+v2/c), min(wl_obs))
    W_r = max(max(wl_obs)/(1+v1/c), max(wl_obs))
    W_a = np.arange(np.log(W_l), np.log(W_r), dv/c)
    # set log wavelength of template spectrum
    index = np.logical_and(W_a >= np.log(min(wl_obs)/(1+v2/c)), W_a <= np.log(max(wl_obs)/(1+v1/c)))
    W = W_a[index]
    # set log wavelength of observed one
    index = np.logical_and(W_a >= np.log(min(wl_obs)), W_a <= np.log(max(wl_obs)))
    w = W_a[index]

    # interpolate flux of either template and observed spectrum to log wavelength
    f = np.interp(w, np.log(wl_obs), flux_obs)
    F = np.interp(W, np.log(wl_temp), flux_temp)
    # do the correlation and normalization
    r = np.correlate(f - np.mean(f) , F - np.mean(F))
    r = r / np.std(f) / np.std(F) / len(w)
    # reset the radial velocity array
    v = dv * np.arange(round((max(w) - max(W)) * c/dv), round((min(w) - min(W)) * c/dv) + 1, 1)
    index = np.logical_and(v >= v1, v <= v2)
    v = v[index]
    r = r[index]
    return v, r
