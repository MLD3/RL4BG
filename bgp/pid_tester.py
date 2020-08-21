import numpy as np
import pandas as pd
import joblib
import os
from joblib import Parallel, delayed
from bgp.rl import pid
from bgp.rl.reward_functions import risk_diff
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
from bgp.evaluation import result_helpers as rh

"""
This script was used to tune the PID and PID-MA baselines. It performs an iterative grid search with exponential
refinement over possible parameters. The best parameters can then be tested using pid_data_collection.py
"""

source_dir = '/source/dir'
data_dir = '/data/dir'
# note: current code assumes different names for residual_bolus and non residual_bolus
name = 'pid_tune_experiment_name'

n_iter = 3
person_grid = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
               ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
               ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
n_days = 10
n_seeds = 3
n_dim = 5
full_save = False
n_jobs = 50
residual_bolus = False

init_grid = {}
best_setting_dict_prev = {}
for key in person_grid:
    init_grid[key] = {'kp': [-1e-5, -1e-4, -1e-3],
                      'ki': [-1e-7, -1e-6, -1e-5],
                      'kd': [-1e-4, -1e-3, -1e-2]}
    best_setting_dict_prev[key] = ((-1e-4, -1e-6, -1e-3), 0)

grid = init_grid

for k in range(n_iter):
    print(k)
    itername = name+'_{}'.format(k)
    if not os.path.exists('{}/{}'.format(data_dir, itername)):
        os.mkdir('{}/{}'.format(data_dir, itername))
    for person in person_grid:
        print('Running person')
        kp_grid = grid[person]['kp']
        ki_grid = grid[person]['ki']
        kd_grid = grid[person]['kd']
        config_arr = []
        env = bgp_env.DeepSACT1DEnv(reward_fun=risk_diff,
                                    patient_name=person,
                                    seeds={'numpy': 0,
                                           'sensor': 0,
                                           'scenario': 0},
                                    reset_lim={'lower_lim': 10, 'upper_lim': 1000},
                                    time=False, meal=False, bw_meals=True,
                                    load=False, gt=False, n_hours=4,
                                    norm=False, time_std=None, action_cap=None, action_bias=0,
                                    action_scale=1, meal_announce=None,
                                    residual_basal=False, residual_bolus=residual_bolus,
                                    residual_PID=False,
                                    fake_gt=False, fake_real=False,
                                    suppress_carbs=False, limited_gt=False,
                                    termination_penalty=None, use_pid_load=False, hist_init=True,
                                    harrison_benedict=True, meal_duration=5, source_dir=source_dir)
        for kp in kp_grid:
            for ki in ki_grid:
                for kd in kd_grid:
                    for seed in range(n_seeds):
                        seed += n_seeds
                        controller = pid.PID(120, kp, ki, kd)
                        config_arr.append({'controller': controller, 'seed': seed})
        res_arr = Parallel(n_jobs=n_jobs)(delayed(pid.pid_test)(env=env,
                                                                pid=config['controller'],
                                                                n_days=n_days,
                                                                seed=config['seed']) for config in config_arr)
        res_grid = {}
        for res in res_arr:
            key = (res['kp'], res['ki'], res['kd'])
            if key not in res_grid:
                res_grid[key] = []
            res_grid[key].append(res['hist'])
        joblib.dump(res_grid, '{}/{}/{}.pkl'.format(RL_DIR, itername, person))

    # generate next grid
    print('Finished running')
    per_patient_perf = []
    for pat in person_grid:
        dat = joblib.load('{}/{}/{}.pkl'.format(RL_DIR, itername, pat))
        for key in dat:
            for seed in range(n_seeds):
                d = {'name': pat, 'kp': key[0], 'ki': key[1], 'kd': key[2], 'seed': seed}
                d['risk'] = dat[key][seed]['Magni_Risk'].mean()
                d['euglycemic'] = np.logical_and(dat[key][seed]['BG'] < 180, dat[key][seed]['BG'] > 70).sum() / len(
                    dat[key][seed]['BG'])
                per_patient_perf.append(d)
    df_pid = pd.DataFrame.from_dict(per_patient_perf)

    best_setting_dict = {}
    for person_name in df_pid['name'].unique():
        print(person_name)
        df_pat = df_pid.query('name == "{}"'.format(person_name))
        best_perf = np.infty
        best_settings = None
        for kp in df_pat['kp'].unique():
            for ki in df_pat['ki'].unique():
                for kd in df_pat['kd'].unique():
                    perf = df_pat.query('kp == {} and ki == {} and kd == {}'.format(kp, ki, kd))['risk']
                    assert len(perf) == n_seeds
                    if perf.mean() < best_perf:
                        best_perf = perf.mean()
                        best_settings = (kp, ki, kd)
        best_setting_dict[person_name] = (best_settings, best_perf)

    k_args = ['kp', 'ki', 'kd']
    patient_grid_dict = {}
    for person_name in df_pid['name'].unique():
        print(person_name)
        patient_grid_dict[person_name] = {'kp': None, 'ki': None, 'kd': None}
        for k_ind in range(3):
            k_type = k_args[k_ind]
            grid = list(np.sort(df_pid.query('name=="{}"'.format(person_name))[k_type].unique())[::-1])
            prev_best = best_setting_dict_prev[person_name][0][k_ind]
            curr_best = best_setting_dict[person_name][0][k_ind]
            perf_grid = []
            for k_val in grid:
                perf_grid.append(
                    df_pid.query('name=="{}" and {}=={}'.format(person_name, k_type, k_val))['euglycemic'].mean())
            patient_grid_dict[person_name][k_type] = rh.update_grid_dict(grid, prev_best, curr_best, n_dim, perf_grid)
    joblib.dump((patient_grid_dict, best_setting_dict), '{}/{}/grid_and_settings.pkl'.format(data_dir, itername))
    grid = patient_grid_dict
    best_setting_dict_prev = best_setting_dict
