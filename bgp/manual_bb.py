import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import namedtuple
import joblib
from joblib import Parallel, delayed
from bgp.rl.reward_functions import risk_diff
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
import bgp.simglucose.controller.basal_bolus_ctrller as bbc

"""
This script runs the BB baseline
"""

Seeds = namedtuple('seeds', ['numpy_seed', 'sensor_seed', 'scenario_seed'])
n_jobs = 50
name = 'basal_bolus'
server = 'mld4'
source_path = '/source/path'
save_path = '/save/path'
full_path = '{}/{}'.format(save_path, name)

if not os.path.exists(full_path):
    os.mkdir(full_path)

def run_bb(name, seed, n_days, full_path):
    q = pd.read_csv('{}/simglucose/params/Quest2.csv'.format(source_path))
    p = pd.read_csv('{}/simglucose/params/vpatient_params.csv'.format(source_path))
    carb_error_mean = 0
    carb_error_std = .2
    carb_miss_prob = .05
    sample_time = 5
    u2ss = p.query('Name=="{}"'.format(name)).u2ss.item()
    bw = p.query('Name=="{}"'.format(name)).BW.item()
    basal = u2ss * bw / 6000
    cr = q.query('Name=="{}"'.format(name)).CR.item()
    cf = q.query('Name=="{}"'.format(name)).CF.item()
    cnt = bbc.ManualBBController(target=140, cr=cr, cf=cf, basal=basal, sample_rate=sample_time,
                                 use_cf=True, use_bol=True, cooldown=180, corrected=True,
                                 use_low_lim=True, low_lim=140)

    res_dict = {}
    env = bgp_env.DeepSACT1DEnv(reward_fun=risk_diff,
                                patient_name=name,
                                seeds={'numpy': seed,
                                       'sensor': seed,
                                       'scenario': seed},
                                reset_lim={'lower_lim': 10, 'upper_lim': 1000},
                                time=False, meal=False, bw_meals=True,
                                load=False, gt=False, n_hours=4,
                                norm=False, time_std=None, action_cap=None, action_bias=0,
                                action_scale=1, meal_announce=None,
                                residual_basal=False, residual_bolus=False,
                                residual_PID=False,
                                fake_gt=False, fake_real=False,
                                suppress_carbs=False, limited_gt=False,
                                termination_penalty=None, hist_init=True, harrison_benedict=True, meal_duration=5,
                                source_path=source_path)
    action = cnt.manual_bb_policy(carbs=0, glucose=140)
    for i in tqdm(range(n_days * int(1440/sample_time))):
        o, r, d, info = env.step(action=action.basal+action.bolus)
        bg = env.env.CGM_hist[-1]
        carbs = info['meal'] * 5
        if np.random.uniform() < carb_miss_prob:
            carbs = 0
        err = np.random.normal(carb_error_mean, carb_error_std)
        carbs = carbs + carbs * err
        action = cnt.manual_bb_policy(carbs=carbs, glucose=bg)
    hist = env.env.show_history()[288:]
    res_dict['person'] = name
    res_dict['seed'] = seed
    res_dict['bg'] = hist['BG'].mean()
    res_dict['risk'] = hist['Risk'].mean()
    res_dict['hyper'] = (hist['BG'] > 180).sum() / len(hist['BG'])
    res_dict['hypo'] = (hist['BG'] < 70).sum() / len(hist['BG'])
    res_dict['event'] = res_dict['hyper'] + res_dict['hypo']
    joblib.dump(hist, '{}/bb_{}_seed{}.pkl'.format(full_path, name, seed))
    return res_dict


patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
            ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
            ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
n_days = 10
seeds = [i for i in range(100)]

settings = itertools.product(patients, seeds)
res_list = Parallel(n_jobs=n_jobs)(delayed(run_bb)(name=s[0], seed=s[1],
                                                   n_days=n_days, full_path=full_path) for s in settings)

