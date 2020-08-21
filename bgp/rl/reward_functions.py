import numpy as np
from bgp.simglucose.analysis.risk import risk_index


def reward_risk(risk_hist, **kwargs):
    return -risk_hist[-1]


def epsilon_risk(risk_hist, **kwargs):
    if risk_hist[-1] < 1:
        return 1
    else:
        return -risk_hist[-1]


def reward_cgm_low_diff(cgm_hist, **kwargs):
    # This is based on a mistake I think I made earlier with reward_risk, it's not actually a good reward function
    if len(cgm_hist) < 2:
        return 0
    return cgm_hist[-2] - cgm_hist[-1]


def reward_cgm_high_diff(cgm_hist, **kwargs):
    # This is based on a mistake I think I made earlier with reward_risk, it's not actually a good reward function
    if len(cgm_hist) < 2:
        return 0
    return -(cgm_hist[-2] - cgm_hist[-1])


def reward_cgm_high(cgm_hist, **kwargs):
    # This is based on a mistake I think I made earlier with reward_risk, it's not actually a good reward function
    return cgm_hist[-1]


def reward_bg_high(bg_hist, **kwargs):
    # This is based on a mistake I think I made earlier with reward_risk, it's not actually a good reward function
    return bg_hist[-1]


def reward_cgm_low(cgm_hist, **kwargs):
    # This is based on a mistake I think I made earlier with reward_risk, it's not actually a good reward function
    return -cgm_hist[-1]


def reward_target(cgm_hist, **kwargs):
    return -1 * np.abs(cgm_hist[-1] - 120)


def risk_diff(cgm_hist, **kwargs):
    if len(cgm_hist) < 2:
        return 0
    _, _, risk_current = risk_index([cgm_hist[-1]], 1)
    _, _, risk_prev = risk_index([cgm_hist[-2]], 1)
    return risk_prev - risk_current


def risk_diff_bg(bg_hist, **kwargs):
    if len(bg_hist) < 2:
        return 0
    _, _, risk_current = risk_index([bg_hist[-1]], 1)
    _, _, risk_prev = risk_index([bg_hist[-2]], 1)
    return risk_prev - risk_current


def risk_bg(bg_hist, **kwargs):
    return -risk_index([bg_hist[-1]], 1)[-1]


def reward_event(cgm_hist, **kwargs):
    if len(cgm_hist) < 1:
        return 0
    if cgm_hist[-1] > 180 or cgm_hist[-1] < 70:
        return -1
    else:
        return 1


def reward_day(cgm_hist, **kwargs):
    cgm_hist = np.array(cgm_hist)
    hypo_cnt = (cgm_hist < 70).sum()
    hyper_cnt = (cgm_hist > 180).sum()
    return -1 * (hypo_cnt * 0.2 + hyper_cnt * 0.1)


def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk


def cameron_reward(bg_hist, **kwargs):
    bg = bg_hist[-1]
    a = .2370  # 1/(mg/dL)
    b = -36.21
    c = 6.0e-5  # (1/(mg/dL)**3)
    d = 177  # mg/dL
    if bg < d:
        risk = a*bg+b+(c*(d-bg)**3)
    else:
        risk = a*bg+b
    return -1*risk
