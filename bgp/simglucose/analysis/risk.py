import numpy as np
import warnings


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = np.array(BG[-horizon:])
        BG_to_compute[BG_to_compute < 1] = 1
        fBG = 1.509 * (np.log(BG_to_compute)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        LBGI = np.nan_to_num(np.mean(rl))
        HBGI = np.nan_to_num(np.mean(rh))
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)

def magni_risk_index(BG, **kwargs):
    bg = max(BG[-1], 1)
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return risk
