# %%
# importing offical module
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import correlate
from astropy import time as T, coordinates as coord, units as u
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
from matplotlib import rcParams
from localreg import *

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 40
# import private module
from norm import norm
from rv_xcor import xcor
from AFS import AFS

# import light speed constant
import scipy.constants
c = scipy.constants.speed_of_light
c = c/1000 # set 'c' is the speed of light (km/s)


#%%
def deres(wl, flux, res_h, res_l):
    lambda0 = np.mean(wl)
    D_lambda = np.mean(np.diff(wl))
    k = lambda0/D_lambda/res_h
    sigma = k * np.sqrt(res_h**2 - res_l**2)/2.355/res_l
    return gaussian_filter1d(flux, sigma, mode = 'reflect')


def macro_v(Teff, logg):
    if(logg >= 3.5): # dwarf: 1984ApJ...281..719G
        macro = 3.95*Teff/1000 - 19.25
        if(macro < 0):
            macro == 0
    elif(logg < 3.5) & (Teff < 5500): # giant: 1982ApJ...262..682G, the Teff of G5 giant is about 5000K
        macro =  7 - (5.5 - Teff/1000)**2.6
    elif(logg < 3.5) & (Teff >= 5500): # giant: 1982ApJ...262..682G
        macro = 7 + (Teff/1000 - 5.5)**2.6
    return macro


def rot_broad(wl, flux, e, vsini):
    lambda0 = np.mean(wl)
    D_lambda = np.mean(np.diff(wl))
    lambda_L = lambda0 * vsini/c

    c1 = 2*(1 - e) / (np.pi*lambda_L*(1 - e/3))
    c2 = e / (2*lambda_L*(1 - e/3))

    lamb = np.arange(-2*np.ceil(lambda_L/D_lambda), 2*np.ceil(lambda_L/D_lambda))*D_lambda + lambda0
    index = (abs(lamb - lambda0) < lambda_L)
    G_rot =  c1*np.sqrt(1 - ((lamb[index]-lambda0)/lambda_L)**2) + c2*(1 - ((lamb[index]-lambda0)/lambda_L)**2)
    return correlate(flux, G_rot/sum(G_rot))


#%%
class mod():
    def __init__(self, fit_file):
        hul = fits.open(fit_file)
        self.Teff, self.logg, self.Feh = np.transpose(hul[0].data)
        self.wl_b = hul[1].data
        self.flux_b = hul[2].data
        self.wl_r = hul[3].data
        self.flux_r = hul[4].data
        self.res_B = 25000
        self.res_R = 25000


    def _dist(self, t, g, m):
        return np.sqrt(((t - self.Teff)/100)**2 + ((g - self.logg)/0.1)**2 + ((m - self.Feh)/0.1)**2)


    def gen_mod(self, Teff, logg, Feh):
        def gen_mat(x, y, z):
            return np.stack((x**3, y**3, z**3, x*y**2, x*z**2, y*x**2, y*z**2, z*x**2, z*y**2, x*y*z, x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, np.ones(len(x))), axis = 1)

        r = self._dist(Teff, logg, Feh)
        index = np.argsort(r)[0:128]
        x = self.Teff[index]
        y = self.logg[index]
        z = self.Feh[index]
        F = self.flux[self.index_sp][index]
        A = gen_mat(x/100, y/0.1, z/0.1)
        coeffs, *_ = np.linalg.lstsq(A, F, rcond=None)
        a = gen_mat(np.array([Teff])/100, np.array([logg])/0.1, np.array([Feh])/0.1)
        return np.dot(a, coeffs)[0] # The length of the array is 1, so we add the index 0.


    def choose_mod(self, Teff, logg, Feh, vsini, res_b, res_r):
        r = self._dist(Teff, logg, Feh)
        mod_flux_b = self.flux_b[np.argmin(r)]
        mod_flux_r = self.flux_r[np.argmin(r)]
        v_mac = macro_v(Teff, logg)
        mod_flux_b = deres(self.wl_b, mod_flux_b, self.res_B, res_b)
        mod_flux_r = deres(self.wl_r, mod_flux_r, self.res_R, res_r)
        mod_v = np.sqrt(vsini**2 + v_mac**2)
        if(mod_v > 0):
            mod_flux_b = rot_broad(self.wl_b, mod_flux_b, 0.6, mod_v)
            mod_flux_r = rot_broad(self.wl_r, mod_flux_r, 0.6, mod_v)
        return mod_flux_b, mod_flux_r


#%%
class obs():
    def __init__(self, d):
        # self.Teff_ref = d['Teff_ref']
        # if(np.isnan(self.Teff_ref)):
        #     self.Teff_ref = -9999
        # self.logg_ref = d['logg_ref']
        # if(np.isnan(self.logg_ref)):
        #     self.logg_ref = -9999
        # self.Feh_ref = d['Feh_ref']
        # if(np.isnan(self.Feh_ref)):
        #     self.Feh_ref = -9999
        # self.vsini_ref = d['vsini_ref']
        # if(np.isnan(self.vsini_ref)):
        #     self.vsini_ref = 0

        self.filename = d['filename']
        self.lmjm = d['lmjm']
        self.snr_B = d['snr_B']
        self.snr_R = d['snr_R']
        self.res_B = d['res_B']
        self.res_R = d['res_R']


    def read_med(self, hul, No_b, No_r):
        flux = hul[No_b].data.field(0)
        invar = hul[No_b].data.field(1)
        wl = hul[No_b].data.field(2)
        pixmask = hul[No_b].data.field(3)
        index_ava = (pixmask == 0) & (invar > 0)
        index_wl = np.argsort(wl[index_ava])
        self.wl_b = wl[index_ava][index_wl][100:-100]
        self.flux_b  = flux[index_ava][index_wl][100:-100]
        self.invar_b = invar[index_ava][index_wl][100:-100]
        index_wl = (self.wl_b >= 4950) & (self.wl_b <= 5320) | (self.wl_b >= 5325) & (self.wl_b <= 5350)
        self.wl_b = self.wl_b[index_wl]
        self.flux_b  = self.flux_b[index_wl]
        self.e_flux_b = 1/np.sqrt(self.invar_b[index_wl])
        self.snr_B = np.median(self.flux_b/self.e_flux_b)

        flux = hul[No_r].data.field(0)
        invar = hul[No_r].data.field(1)
        wl = hul[No_r].data.field(2)
        pixmask = hul[No_r].data.field(3)
        index_ava = (pixmask == 0) & (invar > 0)
        index_wl = np.argsort(wl[index_ava])
        self.wl_r = wl[index_ava][index_wl][100:-100]
        self.flux_r  = flux[index_ava][index_wl][100:-100]
        self.invar_r = invar[index_ava][index_wl][100:-100]
        index_wl = (self.wl_r >= 6300) & (self.wl_r <= 6800)
        self.wl_r = self.wl_r[index_wl]
        self.flux_r  = self.flux_r[index_wl]
        self.e_flux_r = 1/np.sqrt(self.invar_r[index_wl])
        self.snr_R = np.median(self.flux_r/self.e_flux_r)


    def norm_spec(self):
        self.B_flux_b, self.wl_b, self.flux_b, self.index_cr_b, self.index_em_b = norm(self.wl_b, self.flux_b, k = 1/3, upper = 7, lower = 0.5, iter_num = 3)
        self.B_flux_r, self.wl_r, self.flux_r, self.index_cr_r, self.index_em_r = norm(self.wl_r, self.flux_r, k = 1/2, upper = 7, lower = 0.5, iter_num = 3)
        self.wl_b = self.wl_b[self.index_cr_b]
        self.flux_b = self.flux_b[self.index_cr_b]
        self.e_flux_b = (self.e_flux_b/self.B_flux_b)[self.index_cr_b]
        self.wl_r = self.wl_r[self.index_cr_r]
        self.flux_r = self.flux_r[self.index_cr_r]
        self.e_flux_r = (self.e_flux_r/self.B_flux_r)[self.index_cr_r]


    def rv(self, mod_class):
        index_Teff = np.in1d(mod_class.Teff, np.arange(4000, 8100, 1000))
        index_logg = np.in1d(mod_class.logg, np.around(np.arange(1, 5.1, 2)))
        index_Feh = np.in1d(mod_class.Feh, 0)
        index_grid = index_Teff & index_logg & index_Feh
        mod_flux_b = mod_class.flux_b[index_grid]
        mod_flux_r = mod_class.flux_r[index_grid]
        mod_Teff = mod_class.Teff[index_grid]
        mod_logg = mod_class.logg[index_grid]
        # mod_Feh = mod_class.Feh[index_grid]

        rv_b = []
        rv_r = []
        maxCCF_b = []
        maxCCF_r = []
        for i in range(len(mod_flux_b)):
            try:
                mod_flux_B = deres(mod_class.wl_b, mod_flux_b[i], mod_class.res_B, self.res_B)
                v, ccf = xcor(obs = [self.wl_b[self.index_em_b], self.flux_b[self.index_em_b]], temp = [mod_class.wl_b, mod_flux_B], v = [-1000, 1000 ,1])
                rv0_B = v[np.argmax(ccf)]
                r1 = gaussian_filter1d(ccf, sigma = 5, order = 1, mode = 'mirror')
                index1 = (np.diff(r1) < 0) & (r1[1:]*r1[:-1] < 0)
                k = (r1[1:][index1]-r1[:-1][index1]) / (v[1:][index1]-v[:-1][index1])
                rv1_b = v[:-1][index1] - r1[:-1][index1]/k
                rv = rv1_b[np.argmin(abs(rv1_b - rv0_B))]
                rv_b = np.append(rv_b, rv)
                maxCCF_b = np.append(maxCCF_b, np.interp(rv, v, ccf))

                mod_flux_R = deres(mod_class.wl_r, mod_flux_r[i], mod_class.res_R, self.res_R)
                v, ccf = xcor(obs = [self.wl_r[self.index_em_r], self.flux_r[self.index_em_r]], temp = [mod_class.wl_r, mod_flux_R], v = [-1000, 1000 ,1])
                rv0_R = v[np.argmax(ccf)]
                r1 = gaussian_filter1d(ccf, sigma = 5, order = 1, mode = 'mirror')
                index1 = (np.diff(r1) < 0) & (r1[1:]*r1[:-1] < 0)
                k = (r1[1:][index1]-r1[:-1][index1]) / (v[1:][index1]-v[:-1][index1])
                rv1_r = v[:-1][index1] - r1[:-1][index1]/k
                rv = rv1_r[np.argmin(abs(rv1_r - rv0_R))]
                rv_r = np.append(rv_r, rv)
                maxCCF_r = np.append(maxCCF_r, np.interp(rv, v, ccf))
            except:
                print(mod_class.Teff[index_grid][i], mod_class.logg[index_grid][i], mod_class.Feh[index_grid][i])
        index_b = np.argmax(maxCCF_b)
        index_r = np.argmax(maxCCF_r)
        self.rv_B = rv_b[index_b]
        self.rv_R = rv_r[index_r]
        self.Teff_RV_B = mod_Teff[index_b]
        self.Teff_RV_R = mod_Teff[index_r]
        self.logg_RV_B = mod_logg[index_b]
        self.logg_RV_R = mod_logg[index_r]
        self.maxCCF_B = maxCCF_b[index_b]
        self.maxCCF_R = maxCCF_r[index_r]
        self.wl_b = self.wl_b * (1 - self.rv_B/c)
        self.wl_r = self.wl_r * (1 - self.rv_R/c)

        rv_B_mc = []
        rv_R_mc = []
        mod_flux_B = deres(mod_class.wl_b, mod_flux_b[index_b], mod_class.res_B, self.res_B)
        mod_flux_R = deres(mod_class.wl_r, mod_flux_r[index_r], mod_class.res_R, self.res_R)
        rand_mc_b = np.random.randn(100, len(self.wl_b))
        rand_mc_r = np.random.randn(100, len(self.wl_r))
        for i in range(100):
            try:
                flux_mc_b = self.flux_b + rand_mc_b[i] * self.e_flux_b
                v, ccf = xcor(obs = [self.wl_b[self.index_em_b], flux_mc_b[self.index_em_b]], temp = [mod_class.wl_b, mod_flux_B], v = [-100, 100 ,1])
                rv0_B = v[np.argmax(ccf)]
                r1 = gaussian_filter1d(ccf, sigma = 5, order = 1, mode = 'mirror')
                index1 = (np.diff(r1) < 0) & (r1[1:]*r1[:-1] < 0)
                k = (r1[1:][index1]-r1[:-1][index1]) / (v[1:][index1]-v[:-1][index1])
                rv1_b = v[:-1][index1] - r1[:-1][index1]/k
                rv_B_mc = np.append(rv_B_mc, rv1_b[np.argmin(abs(rv1_b - rv0_B))])

                flux_mc_r = self.flux_r + rand_mc_r[i] * self.e_flux_r
                v, ccf = xcor(obs = [self.wl_r[self.index_em_r], flux_mc_r[self.index_em_r]], temp = [mod_class.wl_r, mod_flux_R], v = [-100, 100 ,1])
                rv0_R = v[np.argmax(ccf)]
                r1 = gaussian_filter1d(ccf, sigma = 5, order = 1, mode = 'mirror')
                index1 = (np.diff(r1) < 0) & (r1[1:]*r1[:-1] < 0)
                k = (r1[1:][index1]-r1[:-1][index1]) / (v[1:][index1]-v[:-1][index1])
                rv1_r = v[:-1][index1] - r1[:-1][index1]/k
                rv_R_mc = np.append(rv_R_mc, rv1_r[np.argmin(abs(rv1_r - rv0_R))])
            except:
                print('')

        if(len(rv_B_mc) > 1):
            self.e_rv_B = np.std(rv_B_mc)
        else:
            self.e_rv_B = -9999
        if(len(rv_B_mc) > 1):
            self.e_rv_R = np.std(rv_R_mc)
        else:
            self.e_rv_R = -9999


        index_b = (self.wl_b > 4950) & (self.wl_b < 5350)
        index_r = (self.wl_r > 6300) & (self.wl_r < 6800)
        self.wl_b = self.wl_b[index_b]
        self.flux_b = self.flux_b[index_b]
        self.e_flux_b = self.e_flux_b[index_b]
        self.index_em_b = self.index_em_b[index_b]
        self.wl_r = self.wl_r[index_r]
        self.flux_r = self.flux_r[index_r]
        self.e_flux_r = self.e_flux_r[index_r]
        self.index_em_r = self.index_em_r[index_r]


    def detect_line(self, lamb, D_flux):
        if(lamb > 4900) & (lamb < 5300):
            wl = self.wl_b
            flux = self.flux_b
        elif(lamb > 6300) & (lamb < 6800):
            wl = self.wl_r
            flux = self.flux_r
        else:
            raise ValueError('The wavelength of the line is out of range.')

        index_center = np.argmin(abs(wl - lamb))
        flux_line = flux[index_center]
        flux_HM = (1 + flux_line)/2
        Depth = 1 - flux_line

        if(flux_line < 1 - 2*D_flux):
            flag_line = -1
            flux_thresh = 1 - D_flux
        elif(flux_line >= 1 - 2*D_flux) & (flux_line <= 1 + 2*D_flux) | (abs(wl[index_center] - lamb) > 2):
            flag_line = 0
        elif(flux_line > 1 + 2*D_flux):
            flag_line = 1
            flux_thresh = 1 + D_flux


        if(flag_line == -1):
            for i in range(index_center):
                if(flux[index_center - i] <= flux_HM):
                    index_leftHM = index_center - i
                if(flux[index_center - i] <= flux_thresh):
                    index_leftwing = index_center - i
                if(flux[index_center - i] > flux_thresh):
                    break
            for i in range(len(wl) - index_center):
                if(flux[index_center + i] <= flux_HM):
                    index_rightHM = index_center + i
                if(flux[index_center + i] <= flux_thresh):
                    index_rightwing = index_center + i
                if(flux[index_center + i] > flux_thresh):
                    break

        elif(flag_line == 1):
            for i in range(index_center):
                if(flux[index_center - i] >= flux_HM):
                    index_leftHM = index_center - i
                if(flux[index_center - i] >= flux_thresh):
                    index_leftwing = index_center - i
                if(flux[index_center - i] < flux_thresh):
                    break
            for i in range(len(wl) - index_center):
                if(flux[index_center + i] >= flux_HM):
                    index_rightHM = index_center + i
                if(flux[index_center + i] >= flux_thresh):
                    index_rightwing = index_center + i
                if(flux[index_center + i] < flux_thresh):
                    break

        if(flag_line == 0):
            FWHM = 0
            EW = 0
            index_line = (wl == wl[index_center])
        else:
            index = np.argsort(flux[index_leftHM - 1:index_leftHM + 1])
            wl_leftHM = np.interp(flux_HM, flux[index_leftHM - 1:index_leftHM + 1][index], wl[index_leftHM - 1:index_leftHM + 1][index])
            index = np.argsort(flux[index_rightHM:index_rightHM + 2])
            wl_rightHM = np.interp(flux_HM, flux[index_rightHM:index_rightHM + 2][index], wl[index_rightHM:index_rightHM + 2][index])
            FWHM = wl_rightHM - wl_leftHM
            index = np.argsort(flux[index_leftwing - 1:index_leftwing + 1])
            wl_leftwing = np.interp(flux_thresh, flux[index_leftwing - 1:index_leftwing + 1][index], wl[index_leftwing - 1:index_leftwing + 1][index])
            index = np.argsort(flux[index_rightwing:index_rightwing + 2])
            wl_rightwing = np.interp(flux_thresh, flux[index_rightwing:index_rightwing + 2][index], wl[index_rightwing:index_rightwing + 2][index])
            index_line = (wl >= wl_leftwing) & (wl <= wl_rightwing)
            EW = np.nansum(1/2 * (2 - flux[index_line][:-1] - flux[index_line][1:]) * np.diff(wl[index_line]))


        if(flag_line >= 0) & (lamb > 6560) & (lamb < 6570):
            self.wl_select_r = self.wl_r[self.index_em_r]
            self.flux_select_r = self.flux_r[self.index_em_r]
        elif(flag_line == -1) & (lamb > 6560) & (lamb < 6570):
            index_FWHM = (self.wl_r[self.index_em_r] < wl_leftHM) | (self.wl_r[self.index_em_r] > wl_rightHM)
            self.wl_select_r = self.wl_r[self.index_em_r][index_FWHM]
            self.flux_select_r = self.flux_r[self.index_em_r][index_FWHM]

        return flag_line, Depth, FWHM, EW, index_line


    def interp(self, mod_wl_b, mod_wl_r):
        flux_b = np.nan * np.ones(len(mod_wl_b))
        flux_r = np.nan * np.ones(len(mod_wl_r))

        dw = np.mean(np.diff(mod_wl_b))
        index = np.where(np.diff(self.wl_select_b) > 2*dw)[0]
        index = np.insert(index, [0, len(index)], [-1, len(self.wl_select_b) - 1])
        for i in range(len(index) - 1):
            if(index[i+1] - index[i] > 4):
                index_mod = (mod_wl_b > self.wl_select_b[index[i] + 1]) & (mod_wl_b < self.wl_select_b[index[i+1]])
                flux_b[index_mod] = np.interp(mod_wl_b[index_mod], self.wl_select_b, self.flux_select_b)

        dw = np.mean(np.diff(mod_wl_r))
        index = np.where(np.diff(self.wl_select_r) > 2*dw)[0]
        index = np.insert(index, [0, len(index)], [-1, len(self.wl_select_r) - 1])
        for i in range(len(index) - 1):
            if(index[i+1] - index[i] > 4):
                index_mod = (mod_wl_r > self.wl_select_r[index[i] + 1]) & (mod_wl_r < self.wl_select_r[index[i+1]])
                flux_r[index_mod] = np.interp(mod_wl_r[index_mod], self.wl_select_r, self.flux_select_r)

        self.wl_select_b = mod_wl_b
        self.flux_select_b = flux_b
        self.wl_select_r = mod_wl_r
        self.flux_select_r = flux_r


    def norm_spec2(self, mod_flux_b, mod_flux_r):
        index_b = np.isnan(self.flux_select_b)
        index_r = np.isnan(self.flux_select_r)
        self.B_flux_b, *_ = norm(self.wl_select_b[~index_b], (self.flux_select_b/mod_flux_b)[~index_b], k = 1/3, upper = 3, lower = 0.5, iter_num = 1)
        self.B_flux_r, *_ = norm(self.wl_select_r[~index_r], (self.flux_select_r/mod_flux_r)[~index_r], k = 1/3, upper = 3, lower = 0.5, iter_num = 1)

        self.flux_select_b[~index_b] /= self.B_flux_b
        self.flux_select_r[~index_r] /= self.B_flux_r


    def cal_pars(self, mod_class, Teff_list, logg_list, Feh_list, vsini_list, a_Teff, a_logg, a_Feh, a_vsini, Feh_band = 'B', output = False):
        index_grid = np.in1d(mod_class.Teff, Teff_list) & np.in1d(mod_class.logg, logg_list) & np.in1d(mod_class.Feh, Feh_list)
        mod_flux_b = mod_class.flux_b[index_grid]
        mod_flux_r = mod_class.flux_r[index_grid]
        if(np.count_nonzero(index_grid) == 1):
            mod_flux_b = [mod_flux_b]
            mod_flux_r = [mod_flux_r]

        chi2_b = []
        chi2_r = []
        Teff = []
        logg = []
        Feh = []
        vsini = []
        for v_bot in vsini_list:
            Teff = np.append(Teff, mod_class.Teff[index_grid])
            logg = np.append(logg, mod_class.logg[index_grid])
            Feh = np.append(Feh, mod_class.Feh[index_grid])
            vsini = np.append(vsini, np.ones(len(mod_flux_b)) * v_bot)
            for i in range(len(mod_flux_b)):
                vmac = macro_v(Teff[i], logg[i])
                mod_v = np.sqrt(v_bot**2 + vmac**2)
                mod_flux_B = deres(mod_class.wl_b, mod_flux_b[i], mod_class.res_B, self.res_B)
                mod_flux_R = deres(mod_class.wl_r, mod_flux_r[i], mod_class.res_R, self.res_R)
                mod_flux_B = rot_broad(mod_class.wl_b, mod_flux_B, 0.6, mod_v)
                mod_flux_R = rot_broad(mod_class.wl_r, mod_flux_R, 0.6, mod_v)
                chi2_b = np.append(chi2_b, np.nansum((self.flux_select_b - mod_flux_B)**2 / mod_flux_B))
                chi2_r = np.append(chi2_r, np.nansum((self.flux_select_r - mod_flux_R)**2 / mod_flux_R))
        self.chi2_B = chi2_b
        self.chi2_R = chi2_r
        self.chi2_A = chi2_b * chi2_r

        chi2_Teff = self.chi2_R
        chi2_logg = self.chi2_A
        if(Feh_band == 'B'):
            chi2_Feh = self.chi2_B
        elif(Feh_band == 'R'):
            chi2_Feh = self.chi2_R
        elif(Feh_band == 'A'):
            chi2_Feh = self.chi2_A
        chi2_vsini = self.chi2_A

        index = (chi2_Teff < (1 + a_Teff) * np.min(chi2_Teff))
        index_left = (Teff[index] == np.min(Teff[index]))
        index_right = (Teff[index] == np.max(Teff[index]))
        index = (chi2_Teff <= np.min([np.max(chi2_Teff[index][index_left]), np.max(chi2_Teff[index][index_right])]))
        if(np.max(chi2_Teff[index]) == np.min(chi2_Teff)):
            self.Teff = np.min(Teff[index])
        else:
            W = 1 - 0.8*(chi2_Teff[index] - np.min(chi2_Teff))/(np.max(chi2_Teff[index]) - np.min(chi2_Teff))
            self.Teff = np.average(Teff[index], weights = W)
        if(np.min(Teff[index]) <= np.min(Teff_list)):
            self.Teff_left = np.min(Teff_list)
        else:
            self.Teff_left = Teff_list[np.searchsorted(Teff_list, np.min(Teff[index]), side = 'left') - 1]
        if(np.max(Teff[index]) >= np.max(Teff_list)):
            self.Teff_right = np.max(Teff_list)
        else:
            self.Teff_right = Teff_list[np.searchsorted(Teff_list, np.max(Teff[index]), side = 'right')]

        index = (chi2_logg < (1 + a_logg) * np.min(chi2_logg))
        index_left = (logg[index] == np.min(logg[index]))
        index_right = (logg[index] == np.max(logg[index]))
        index = (chi2_logg <= np.min([np.max(chi2_logg[index][index_left]), np.max(chi2_logg[index][index_right])]))
        if(np.max(chi2_logg[index]) == np.min(chi2_logg)):
            self.logg = np.min(logg[index])
        else:
            W = 1 - 0.8*(chi2_logg[index] - np.min(chi2_logg))/(np.max(chi2_logg[index]) - np.min(chi2_logg))
            self.logg = np.average(logg[index], weights = W)
        if(np.min(logg[index]) <= np.min(logg_list)):
            self.logg_left = np.min(logg_list)
        else:
            self.logg_left = logg_list[np.searchsorted(logg_list, np.min(logg[index]), side = 'left') - 1]
        if(np.max(logg[index]) >= np.max(logg_list)):
            self.logg_right = np.max(logg_list)
        else:
            self.logg_right = logg_list[np.searchsorted(logg_list, np.max(logg[index]), side = 'right')]

        index = (chi2_Feh < (1 + a_Feh) * np.min(chi2_Feh))
        index_left = (Feh[index] == np.min(Feh[index]))
        index_right = (Feh[index] == np.max(Feh[index]))
        index = (chi2_Feh <= np.min([np.max(chi2_Feh[index][index_left]), np.max(chi2_Feh[index][index_right])]))
        if(np.max(chi2_Feh[index]) == np.min(chi2_Feh)):
            self.Feh = np.min(Feh[index])
        else:
            W = 1 - 0.8*(chi2_Feh[index] - np.min(chi2_Feh))/(np.max(chi2_Feh[index]) - np.min(chi2_Feh))
            self.Feh = np.average(Feh[index], weights = W)
        if(np.min(Feh[index]) <= np.min(Feh_list)):
            self.Feh_left = np.min(Feh_list)
        else:
            self.Feh_left = Feh_list[np.searchsorted(Feh_list, np.min(Feh[index]), side = 'left') - 1]
        if(np.max(Feh[index]) >= np.max(Feh_list)):
            self.Feh_right = np.max(Feh_list)
        else:
            self.Feh_right = Feh_list[np.searchsorted(Feh_list, np.max(Feh[index]), side = 'right')]

        index = (chi2_vsini < (1 + a_vsini) * np.min(chi2_vsini))
        index_left = (vsini[index] == np.min(vsini[index]))
        index_right = (vsini[index] == np.max(vsini[index]))
        index = (chi2_vsini <= np.min([np.max(chi2_vsini[index][index_left]), np.max(chi2_vsini[index][index_right])]))
        if(np.max(chi2_vsini[index]) == np.min(chi2_vsini)):
            self.vsini = np.min(vsini[index])
        else:
            W = 1 - 0.8*(chi2_vsini[index] - np.min(chi2_vsini))/(np.max(chi2_vsini[index]) - np.min(chi2_vsini))
            self.vsini = np.average(vsini[index], weights = W)
        if(np.min(vsini[index]) <= np.min(vsini_list)):
            self.vsini_left = np.min(vsini_list)
        else:
            self.vsini_left = vsini_list[np.searchsorted(vsini_list, np.min(vsini[index]), side = 'left') - 1]
        if(np.max(vsini[index]) >= np.max(vsini_list)):
            self.vsini_right = np.max(vsini_list)
        else:
            self.vsini_right = vsini_list[np.searchsorted(vsini_list, np.max(vsini[index]), side = 'right')]

        if(output):
            return Teff, logg, Feh, vsini, chi2_b, chi2_r



#%%
def main(i, d, mod_class):
    fit_filename = '/data/dr9/v1.0/medfits/' + d['filename'].loc[i]
    with fits.open(fit_filename) as hul:
        extname = np.transpose(np.array(hul.info(output = False), dtype = object))[1]

        for j in range(dd['counts'].loc[i]): # must use method 'loc'
            obs_spec = obs(d.iloc[i+j]) # set obs_spec as class obs
            # if(obs_spec.Teff_ref == -9999) | (obs_spec.logg_ref == -9999) | (obs_spec.Feh_ref == -9999):
                # continue
            index_No_B = np.in1d(extname, 'B-' + str(obs_spec.lmjm))
            index_No_R = np.in1d(extname, 'R-' + str(obs_spec.lmjm))
            if(np.count_nonzero(index_No_B) == 0) or (np.count_nonzero(index_No_R) == 0):
                print('No spectrum!')
                print(obs_spec.filename)
                print(str(obs_spec.lmjm))
                continue
            else:
                No_B = np.arange(len(extname))[index_No_B][0]
                No_R = np.arange(len(extname))[index_No_R][0]
            obs_spec.read_med(hul, No_B, No_R)

            if(len(obs_spec.wl_b) < 3200) or (len(obs_spec.wl_r) < 3200):
                print('Too many bad pixels!')
                print(obs_spec.filename)
                print(str(obs_spec.lmjm))
                continue

            obs_spec.flag = 0
            ######
            obs_spec.norm_spec()
            obs_spec.rv(mod_class)
            if(abs(obs_spec.rv_B - obs_spec.rv_R) > 10):
                obs_spec.flag += 16

            flag_Ha, Depth_Ha, FWHM_Ha, EW_Ha, Ha_index_line = obs_spec.detect_line(6564.6, 0.02)
            flag_Li, Depth_Li, FWHM_Li, EW_Li, Li_index_line = obs_spec.detect_line(6709.7, 0.02)
            if(flag_Ha == 1):
                obs_spec.flag += 8

            obs_spec.wl_select_b = obs_spec.wl_b[obs_spec.index_em_b]
            obs_spec.flux_select_b = obs_spec.flux_b[obs_spec.index_em_b]
            obs_spec.interp(mod_class.wl_b, mod_class.wl_r)

            ######
            Teff_list = np.arange(3500, 12001, 500)
            logg_list = np.arange(0, 5.1, 1)
            Feh_list = np.around(np.append(np.arange(-4.0, -1.0, 0.6), np.arange(-1.0, 0.6, 0.5)), 1)
            # vsini_list = np.arange(0, 241, 40)
            obs_spec.cal_pars(mod_class,
                                Teff_list,
                                logg_list,
                                Feh_list,
                                [0],
                                a_Teff = 0.3,
                                a_logg = 0.3,
                                a_Feh = 0.3,
                                a_vsini = 1,
                                Feh_band = 'B',
                                output = False)

            ######
            mod_flux_b, mod_flux_r = mod_class.choose_mod(obs_spec.Teff, obs_spec.logg, obs_spec.Feh, obs_spec.vsini, obs_spec.res_B, obs_spec.res_R)
            mod_flux_b += np.random.randn(len(mod_flux_b))/obs_spec.snr_B
            mod_flux_r += np.random.randn(len(mod_flux_r))/obs_spec.snr_R
            obs_spec.norm_spec2(mod_flux_b, mod_flux_r)

            ######
            Teff_list = np.arange(3500, 12001, 500)
            logg_list = np.around(np.arange(0, 5.1, 0.5), 1)
            Feh_list = np.around(np.append(np.arange(-3.8, -1.0, 0.4), np.arange(-1.0, 0.6, 0.3)), 1)
            vsini_list = np.arange(0, 241, 40)

            obs_spec.cal_pars(mod_class,
                                Teff_list[(Teff_list >= obs_spec.Teff_left) & (Teff_list <= obs_spec.Teff_right)],
                                logg_list[(logg_list >= obs_spec.logg_left - 0.5) & (logg_list <= obs_spec.logg_right + 0.5)],
                                Feh_list[(Feh_list >= obs_spec.Feh_left - 0.5) & (Feh_list <= obs_spec.Feh_right + 0.5)],
                                vsini_list,
                                a_Teff = 0.3,
                                a_logg = 0.3,
                                a_Feh = 0.3,
                                a_vsini = 1,
                                Feh_band = 'B',
                                output = False)

            ######
            mod_flux_b, mod_flux_r = mod_class.choose_mod(obs_spec.Teff, obs_spec.logg, obs_spec.Feh, obs_spec.vsini, obs_spec.res_B, obs_spec.res_R)
            mod_flux_b += np.random.randn(len(mod_flux_b))/obs_spec.snr_B
            mod_flux_r += np.random.randn(len(mod_flux_r))/obs_spec.snr_R
            obs_spec.norm_spec2(mod_flux_b, mod_flux_r)

            ######
            Teff_list = np.append(np.arange(3500, 7500, 200), np.arange(7500, 12001, 250))
            logg_list = np.around(np.arange(0, 5.1, 0.25), 2)
            Feh_list = np.around(np.append(np.arange(-4, -0.4, 0.2), np.arange(-0.4, 0.6, 0.1)), 1)
            # vsini_list = np.arange(0, 241, 40)
            vsini_list = np.array([0, 10, 20, 30, 40, 60, 80, 100, 140, 180, 240])

            obs_spec.cal_pars(mod_class,
                                Teff_list[(Teff_list >= obs_spec.Teff_left) & (Teff_list <= obs_spec.Teff_right)],
                                logg_list[(logg_list >= obs_spec.logg_left - 0.5) & (logg_list <= obs_spec.logg_right + 0.5)],
                                Feh_list[(Feh_list >= obs_spec.Feh_left - 0.5) & (Feh_list <= obs_spec.Feh_right + 0.5)],
                                vsini_list[(vsini_list >= obs_spec.vsini_left - 40) & (vsini_list <= obs_spec.vsini_right + 40)],
                                a_Teff = 0.3,
                                a_logg = 0.3,
                                a_Feh = 0.3,
                                a_vsini = 1,
                                Feh_band = 'A',
                                output = False)
            Teff2 = obs_spec.Teff
            logg2 = obs_spec.logg
            Feh2 = obs_spec.Feh
            vsini2 = obs_spec.vsini

            if(obs_spec.Teff < 4000) | (obs_spec.Teff > 7000):
                obs_spec.flag += 1


            ######
            mod_flux_b_final, mod_flux_r_final = mod_class.choose_mod(obs_spec.Teff, obs_spec.logg, obs_spec.Feh, obs_spec.vsini, obs_spec.res_B, obs_spec.res_R)
            chi2_final_B = np.nansum((obs_spec.flux_select_b - mod_flux_b_final)**2 / mod_flux_b_final)
            chi2_final_R = np.nansum((obs_spec.flux_select_r - mod_flux_r_final)**2 / mod_flux_r_final)
            if(chi2_final_B > 40):
                obs_spec.flag += 4
            if(chi2_final_R > 10):
                obs_spec.flag += 2


            ######
            with open('par_test.csv', 'a') as item:
                item.write(d['uid'].iloc[i+j] + ',')
                item.write(str('%.6f'%d['RA'].iloc[i+j]) + ',')
                item.write(str('%.6f'%d['DEC'].iloc[i+j]) + ',')

                item.write(d['filename'].iloc[i+j] + ',')
                item.write(d['planid'].iloc[i+j] + ',')
                item.write(str('%02d'%d['spid'].iloc[i+j]) + ',')
                item.write(str('%03d'%d['fiberid'].iloc[i+j]) + ',')
                item.write(str('%d'%d['lmjd'].iloc[i+j]) + ',')
                item.write(str('%d'%d['lmjm'].iloc[i+j]) + ',')
                item.write(str('%.6f'%d['BJD'].iloc[i+j]) + ',')
                item.write(str('%.1f'%d['snr_B'].iloc[i+j]) + ',')
                item.write(str('%.1f'%d['snr_R'].iloc[i+j]) + ',')
                item.write(str('%d'%d['res_B'].iloc[i+j]) + ',')
                item.write(str('%d'%d['res_R'].iloc[i+j]) + ',')

                item.write(str('%d'%obs_spec.flag) + ',')

                item.write(str('%.1f'%obs_spec.rv_B) + ',')
                item.write(str('%.2f'%obs_spec.e_rv_B) + ',')
                item.write(str('%d'%obs_spec.Teff_RV_B) + ',')
                item.write(str('%.2f'%obs_spec.logg_RV_B) + ',')
                item.write(str('%.2f'%0) + ',')
                item.write(str('%.4f'%obs_spec.maxCCF_B) + ',')

                item.write(str('%.1f'%obs_spec.rv_R) + ',')
                item.write(str('%.2f'%obs_spec.e_rv_R) + ',')
                item.write(str('%d'%obs_spec.Teff_RV_B) + ',')
                item.write(str('%.2f'%obs_spec.logg_RV_B) + ',')
                item.write(str('%.2f'%0) + ',')
                item.write(str('%.4f'%obs_spec.maxCCF_R) + ',')

                item.write(str('%d'%flag_Ha) + ',')
                item.write(str('%.4f'%FWHM_Ha) + ',')
                item.write(str('%.4f'%Depth_Ha) + ',')
                item.write(str('%.4f'%EW_Ha) + ',')

                item.write(str('%d'%flag_Li) + ',')
                item.write(str('%.4f'%FWHM_Li) + ',')
                item.write(str('%.4f'%Depth_Li) + ',')
                item.write(str('%.4f'%EW_Li) + ',')

                item.write(str('%d'%Teff2) + ',')
                item.write(str('%.2f'%logg2) + ',')
                item.write(str('%.2f'%Feh2) + ',')
                item.write(str('%d'%vsini2) + ',')

                item.write(str('%.4f'%chi2_final_B) + ',')
                item.write(str('%.4f'%chi2_final_R) + '\n')


#%%
if __name__ == '__main__':
    with open('./par_test.csv', 'w') as item:
        item.write(
            'uid,RA,DEC,' +\
            'filename,planid,spid,fiberid,lmjd,lmjm,BJD,snr_B,snr_R,res_B,res_R,' +\
            'flag,' +\
            'RV_B,e_RV_B,Teff_RV_B,logg_RV_B,Feh_RV_B,maxCCF_B,' +\
            'RV_R,e_RV_R,Teff_RV_R,logg_RV_R,Feh_RV_R,maxCCF_R,' +\
            'flag_Ha,FWHM_Ha,Depth_Ha,EW_Ha,' +\
            'flag_Li,FWHM_Li,Depth_Li,EW_Li,' +\
            'Teff,logg,Feh,vsini,chi2_final_B,chi2_final_R\n'
        )

    mod0_spec = mod('/data/Model/SPECTRUM/mod_normB+nR_25000_01_vmic+03.fits')

    # d = pd.read_csv('lamost_med_dr9_BR_snr10_res_rv1xAPOGEE_DR17.csv')
    # d = pd.read_csv('lamost_med_dr9_BR_snr10_res_rv1xGALAH_DR3.csv')
    # d = pd.read_csv('lamost_med_dr9_BR_snr100_res_rv1xAPOGEE_DR17.csv')
    # d = pd.read_csv('lamost_med_dr9_BR_snr50_res_rv1xGALAH_DR3.csv')
    # d = pd.read_csv('dr9_v1.0_MRS_BR_snr20_res_rv1.csv')
    # d = pd.read_csv('dr9_v1.0_MRS_BR_snr10_20_res_rv1.csv')
    d = pd.read_csv('test_input.csv')

    # for n in tqdm(range()):
    #     os.system('rm par_result/par_snr10_20_' + str('%07d'%(10000*n)) + '.csv')
    #     df = d[10000*n:10000*(n+1)].sort_values(by = 'filename', ignore_index = True)
    #     # df = d.sort_values(by = 'filename', ignore_index = True)
    #     filename_counts = df['filename'].value_counts(sort = False)
    #     idx = np.zeros(len(filename_counts)).astype(int)
    #     for i in range(len(filename_counts) - 1):
    #         idx[i + 1] = idx[i] + filename_counts.to_numpy()[i]
    #     dd = pd.DataFrame(data = np.transpose([filename_counts.index.to_numpy(), filename_counts.to_numpy()]), columns = ['fileName', 'counts'], index = idx)
    #     # for i in tqdm(dd.index):
    #         # main(i, n, df, mod0_spec)
    #     Parallel(n_jobs = 90, verbose = True)(delayed(main)(i, n, df, mod0_spec) for i in tqdm(dd.index))

    df = d.sort_values(by = 'filename', ignore_index = True)
    filename_counts = df['filename'].value_counts(sort = False)
    idx = np.zeros(len(filename_counts)).astype(int)
    for i in range(len(filename_counts) - 1):
        idx[i + 1] = idx[i] + filename_counts.to_numpy()[i]
    dd = pd.DataFrame(data = np.transpose([filename_counts.index.to_numpy(), filename_counts.to_numpy()]), columns = ['fileName', 'counts'], index = idx)

    # for i in tqdm(dd.index):
        # main(i, df, mod0_spec)
    Parallel(n_jobs = 90, verbose = True)(delayed(main)(i, df, mod0_spec) for i in tqdm(dd.index))

