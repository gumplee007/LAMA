def air2vac(wl):
    """
    Convert wavelength from air to vaccum.
    The formula is derived by N. Piskunov and included in VALD3 tools.
    refs: Piskunov et al. 1995A&AS..112..525P
    """
    s = (10**4) / wl
    fact = 1 + 8.336624212083e-5 + 2.408926869968e-2 / (130.1065924522 - s**2) + 1.599740894897e-4 / (38.92568793293 - s**2)
    
    return fact * wl