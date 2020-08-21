import pandas as pd
import joblib
import os
from joblib import Parallel, delayed
from bgp.rl import pid
from bgp.rl.reward_functions import risk_diff
import bgp.simglucose.envs.simglucose_gym_env as bgp_env

"""
This script was used to test the PID and PID-MA baselines. Note that it assumes 
"""

data_dir = '/' # '/data/dir'
source_dir = '/home/ifox/BGP_MLHC_trim'  # '/source/dir'
name = 'experiment_name'
save_dir = '{}/{}'.format(data_dir, name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
person_options = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
n_days = 10
n_seeds = 100
seed_offset = 1234
full_save = False
residual_bolus = False
tstd_options = [0.1, 1, 10]
carb_error_std = 0
carb_miss_prob = 0
# These names are determined by name in pid_test.py
if residual_bolus:
    grid, settings = joblib.load('/rl/dir/pid_ma_tune_experiment_name_final_itr/grid_and_settings.pkl')
else:
    grid, settings = joblib.load('/rl/dir/pid_tune_experiment_name_final_itr/grid_and_settings.pkl')

n_jobs = 50
v_params = pd.read_csv('/source/path/simglucose/params/vpatient_params.csv')
for tstd in tstd_options:
    for person in person_options:
        vp_row = v_params.query('Name=="{}"'.format(person))
        basal = (vp_row.u2ss*vp_row.BW/6000).squeeze()
        config_arr = []
        for seed in range(n_seeds):
            kp, ki, kd = settings[person][0]
            use_seed = seed + seed_offset
            controller = pid.PID(140, kp, ki, kd, basal=basal)
            config_arr.append({'controller': controller, 'seed': use_seed})
        env = bgp_env.DeepSACT1DEnv(reward_fun=risk_diff,
                                    patient_name=person,
                                    seeds={'numpy': 0,
                                           'sensor': 0,
                                           'scenario': 0},
                                    reset_lim={'lower_lim': 10, 'upper_lim': 1000},
                                    time=False, meal=False, bw_meals=True,
                                    load=False, gt=False, n_hours=4,
                                    norm=False, time_std=tstd, action_cap=None, action_bias=0,
                                    action_scale=1, meal_announce=None,
                                    residual_basal=False, residual_bolus=residual_bolus,
                                    residual_PID=False,
                                    fake_gt=False, fake_real=False,
                                    suppress_carbs=False, limited_gt=False,
                                    termination_penalty=None, use_pid_load=False, hist_init=True,
                                    harrison_benedict=True, meal_duration=5,
                                    carb_error_std=carb_error_std, carb_miss_prob=carb_miss_prob)
        res_arr = Parallel(n_jobs=n_jobs)(delayed(pid.pid_test)(env=env,
                                                                pid=config['controller'],
                                                                n_days=n_days,
                                                                seed=config['seed'],
                                                                full_save=full_save) for config in config_arr)
        joblib.dump(res_arr, '{}/{}_simulation_tstd{}.pkl'.format(save_dir, person, tstd))
