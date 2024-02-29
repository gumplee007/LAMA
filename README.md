# LAMA
LAMA: LAMOST Medium Resolution Spectral Analysis Pipeline
Author: Chunqian Li (lcq@nao.cas.cn)
Date: 2024/2/29

LAMA is used to estimate the stellar parameters, including Teff, logg, [Fe/H], RV and vsini from LAMOST-MRS spectrum.

LAMA.py: the main code of LAMA
air2vac.py: a function to convert the wavelength from air to vaccum.
mod2fits.py: generates a fits file from the binary files of templates.
norm.py: normalizes the spectrum.
rv_xcor.py: calculates the Cross-Correlation Function between observed and template spectra.
