import ast
import os
import joblib
import json
import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from bgp.rl.rlkit_platform import simulate_policy, get_best_itr
from bgp.rl import reward_functions

BGP_DIR = '/bgp/dir'

def process_filename(file, splitlast):
    """
    Extracts variable parameters from filename scheme
    """
    processed = {}
    characteristics = file.split(';')[:-1]
    if splitlast:
        characteristics += [file.split(';')[-1].split('.')[0]]
    processed['experiment_name'] = characteristics[0]
    for i in range(1, len(characteristics)):
        key, val = characteristics[i].split('=')
        try:
            processed[key] = ast.literal_eval(val)
        except:
            # messy
            processed[key] = val
    return processed

def rlkit_load_it_up(path, itr=None, n_jobs=10, run_if_absent=False, replace_if_present=False, save_q=False,
                     cpu_only=False, gpu_remap=None, new_eval=True, bgp_dir='/bgp/dir', eval_seed=None,
                     n_eval_runs=None, night_only=False, meal_only=False, n_eval_start=0, n_days=10, suffix='',
                     use_min=True):
    """
    A helper function to load up a dataframe containing result summaries from a run
    """
    directory = '{}/rl_res/{}'.format(bgp_dir, path)
    files = os.listdir(directory)
    files = list(filter(lambda x: '_' in x, files))
    list_files = Parallel(n_jobs=n_jobs, verbose=1)(delayed(rlkit_load_and_summarize)(directory, f, itr, run_if_absent,
                                                                                      replace_if_present, save_q,
                                                                                      cpu_only, gpu_remap, new_eval,
                                                                                      bgp_dir, eval_seed, n_eval_runs,
                                                                                      night_only, meal_only,
                                                                                      n_eval_start, n_days,
                                                                                      suffix, use_min) for f in files)
    if n_eval_runs is not None:
        list_files_exp = []
        for l in list_files:
            list_files_exp += l
        list_files = list_files_exp
    list_files = list(filter(lambda x: len(x) > 0, list_files))
    df = pd.DataFrame.from_dict(list_files)
    # get variables that change over run
    variable_grid = {}
    for fl in files:
        processed = process_filename(file=fl, splitlast=False)
        for key in processed:
            if key not in variable_grid:
                variable_grid[key] = set({})
            variable_grid[key].add(processed[key])
    variable_grid_nontrivial = {k: v for k, v in variable_grid.items() if len(v) > 1}
    return df, variable_grid_nontrivial


def old_variant_update(variant, bgp_dir):
    """
    This function is used to allow loading old variant schemas after code updates. It inserts default values for new
    parameters
    """
    data_dir = bgp_dir.split('/')[1]
    log_dir_arr = variant['log_dir'].split('/')
    log_dir_arr[1] = data_dir
    log_dir = '/'.join(log_dir_arr)
    variant['log_dir'] = log_dir
    if 'n_sim_days' not in variant:
        variant['n_sim_days'] = 10
    if 'sim_seed_mod' not in variant:
        variant['sim_seed_mod'] = 1
    if 'model_type' not in variant:
        variant['model_type'] = 'dqn'
    if 'include_time' not in variant:
        variant['include_time'] = False
    if 'include_meal' not in variant:
        variant['include_meal'] = False
    if 'action_dim' not in variant:
        variant['action_dim'] = 'old'
    if 'use_ground_truth' not in variant:
        variant['use_ground_truth'] = False
    if 'net_size' not in variant:
        variant['net_size'] = 32
    if 'reset_lim' not in variant:
        variant['reset_lim'] = {'lower_lim': 20, 'upper_lim': 500}
    if 'bw_meals' not in variant:
        variant['bw_meals'] = False
    if 'norm' not in variant:
        variant['norm'] = False
    if 'time_std' not in variant:
        variant['time_std'] = None
    if 'action_repeat' not in variant:
        variant['action_repeat'] = 1
    if 'load' not in variant:
        variant['load'] = True
    if 'use_pid_load' not in variant:
        variant['use_pid_load'] = False
    if 'hist_init' not in variant:
        variant['hist_init'] = False
    if 'use_old_patient_env' not in variant:
        variant['use_old_patient_env'] = True
    if 'action_cap' not in variant:
        variant['action_cap'] = 0.1
    if 'action_bias' not in variant:
        variant['action_bias'] = 0
    if 'action_scale' not in variant:
        variant['action_scale'] = 1
    if 'meal_announce' not in variant:
        variant['meal_announce'] = None
    if 'residual_basal' not in variant:
        variant['residual_basal'] = False
    if 'residual_bolus' not in variant:
        variant['residual_bolus'] = False
    if 'residual_PID' not in variant:
        variant['residual_PID'] = False
    if 'fake_gt' not in variant:
        variant['fake_gt'] = False
    if 'fake_real' not in variant:
        variant['fake_real'] = False
    if 'suppress_carbs' not in variant:
        variant['suppress_carbs'] = False
    if 'limited_gt' not in variant:
        variant['limited_gt'] = False
    if 'termination_penalty' not in variant:
        variant['termination_penalty'] = None
    if 'warm_start' not in variant:
        variant['warm_start'] = None
    if 'weekly' not in variant:
        variant['weekly'] = None
    if 'update_seed_on_reset' not in variant:
        variant['update_seed_on_reset'] = False
    if 'num_eval_runs' not in variant:
        variant['num_eval_runs'] = 1
    if 'deterministic_meal_size' not in variant:
        variant['deterministic_meal_size'] = False
    if 'deterministic_meal_time' not in variant:
        variant['deterministic_meal_time'] = False
    if 'deterministic_meal_occurrence' not in variant:
        variant['deterministic_meal_occurrence'] = False
    if 'basal_scaling' not in variant:
        variant['basal_scaling'] = 43.2
    if 'deterministic_init' not in variant:
        variant['deterministic_init'] = False
    if 'harrison_benedict_sched' not in variant:
        variant['harrison_benedict_sched'] = False
    if 'unrealistic' not in variant:
        variant['unrealistic'] = False
    if 'restricted_sched' not in variant:
        variant['restricted_sched'] = False
    if 'meal_duration' not in variant:
        variant['meal_duration'] = 1
    if 'independent_init' not in variant:
        variant['independent_init'] = None
    if 'rolling_insulin_lim' not in variant:
        variant['rolling_insulin_lim'] = None
    if 'universal' not in variant:
        variant['universal'] = False
    if 'finish_mod' not in variant:
        variant['finish_mod'] = None
    if 'reward_bias' not in variant:
        variant['reward_bias'] = 0
    if 'finish_iter' not in variant:
        variant['finish_itr'] = 'best'
    if 'use_min' not in variant:
        variant['use_min'] = True
    if 'carb_error_std' not in variant:
        variant['carb_error_std'] = 0
    if 'carb_miss_prob' not in variant:
        variant['carb_miss_prob'] = 0
    return variant


def rlkit_evaluate(directory, fl, itr, save_q, cpu_only, gpu_remap, new_eval, bgp_dir, submod, use_min):
    """
    This is a helper function to perform evaluation when loading results for evaluations that failed (or when
    we wish to change evaluation schemes)
    """
    print('evaluating')
    with open('{}/{}/variant.json'.format(directory, fl)) as f:
        variant = json.load(f)
    variant = old_variant_update(variant, bgp_dir=bgp_dir)
    simulate = simulate_policy(variant, itr, save_q, cpu_only=cpu_only, gpu_remap=gpu_remap, new_eval=new_eval,
                               sim_seed_submod=submod, use_min=use_min)
    if save_q:
        simulate, q_save, adv_save = simulate
        if itr is None:
            joblib.dump((q_save, adv_save), '{}/{}/q_and_adv.pkl'.format(directory, fl))
        else:
            joblib.dump((q_save, adv_save), '{}/{}/q_and_adv_{}.pkl'.format(directory, fl, itr))
    if itr is None:
        joblib.dump(simulate, '{}/{}/simulate.pkl'.format(directory, fl))
    else:
        if submod is not None:
            if submod == 0:
                # some backwards compatibility
                joblib.dump(simulate, '{}/simulate_best.pkl'.format(variant['log_dir']))
            joblib.dump(simulate, '{}/simulate_best_{}.pkl'.format(variant['log_dir'], submod))
        else:
            joblib.dump(simulate, '{}/{}/simulate_{}.pkl'.format(directory, fl, itr))


def get_res_dict(simulate):
    res = {}
    # mag_risk
    risk_arr = []
    bg = simulate['BG']
    for j in range(len(bg)):
        risk_arr.append(-1 * reward_functions.magni_reward([max(bg.values[j], 1)]))
    res['mean_mag_risk'] = np.mean(risk_arr)
    res['mean_risk'] = np.mean(simulate['Risk'])
    res['mean_lbgi'] = np.mean(simulate['LBGI'])
    res['mean_hbgi'] = np.mean(simulate['HBGI'])
    res['hist_risk'] = np.histogram(simulate['Risk'])
    res['mean_bg'] = np.mean(simulate['BG'])
    res['min_bg'] = np.min(simulate['BG'])
    res['max_bg'] = np.max(simulate['BG'])
    res['mean_insulin'] = np.mean(simulate['insulin'])
    hypo_percent = (simulate['BG'] < 70).sum() / len(simulate['BG'])
    hyper_percent = (simulate['BG'] > 180).sum() / len(simulate['BG'])
    res['hypo'] = hypo_percent
    res['hyper'] = hyper_percent
    res['event'] = hypo_percent + hyper_percent
    # detect collapse
    res['fail_index'] = np.inf
    if min(simulate['BG']) < 10:
        res['fail_index'] = np.where(simulate['BG'] < 10)[0][0]
    idx_bool = simulate['CHO'].rolling(window=12).max().fillna(0) > 0
    ins = simulate['insulin'].sum()
    meal_ins = simulate[idx_bool]['insulin'].sum()
    res['insulin'] = ins
    res['meal_ins'] = meal_ins/ins
    return res

def _rlkit_load_and_summarize(directory, fl, itr, run_if_absent, replace_if_present, save_q, cpu_only, gpu_remap,
                             new_eval, bgp_dir, eval_seed, submod, night_only, meal_only, n_days, suffix, use_min):
    res = process_filename(fl, splitlast=False)
    with open('{}/{}/variant.json'.format(directory, fl)) as f:
        variant = json.load(f)
    variant = old_variant_update(variant, bgp_dir)
    if itr is None:
        if not os.path.exists('{}/{}/simulate.pkl'.format(directory, fl)) and run_if_absent:
            rlkit_evaluate(directory, fl, itr, save_q, cpu_only=cpu_only, gpu_remap=gpu_remap,
                           new_eval=new_eval, bgp_dir=bgp_dir, submod=submod, use_min=use_min)
        elif replace_if_present:
            rlkit_evaluate(directory, fl, itr, save_q, cpu_only=cpu_only, gpu_remap=gpu_remap,
                           new_eval=new_eval, bgp_dir=bgp_dir, submod=submod, use_min=use_min)
    else:
        if not os.path.exists('{}/{}/simulate_{}{}.pkl'.format(directory, fl, itr, suffix)) and run_if_absent:
            rlkit_evaluate(directory, fl, itr, save_q, cpu_only=cpu_only, gpu_remap=gpu_remap,
                           new_eval=new_eval, bgp_dir=bgp_dir, submod=submod, use_min=use_min)
        elif replace_if_present:
            rlkit_evaluate(directory, fl, itr, save_q, cpu_only=cpu_only, gpu_remap=gpu_remap,
                           new_eval=new_eval, bgp_dir=bgp_dir, submod=submod, use_min=use_min)
    try:
        progress = pd.read_csv('{}/{}/progress.csv'.format(directory, fl))
        if itr is None:
            simulate = joblib.load('{}/{}/simulate.pkl'.format(directory, fl))
        else:
            if eval_seed is not None:
                simulate = joblib.load('{}/{}/simulate_rerun_seed{}.pkl'.format(directory, fl, eval_seed))
            else:
                if submod is not None:
                    res['submod'] = submod
                    simulate = joblib.load('{}/{}/simulate_{}_{}{}.pkl'.format(directory, fl, itr, submod, suffix))
                else:
                    simulate = joblib.load('{}/{}/simulate_{}.pkl'.format(directory, fl, itr))
        if simulate is -1:
            return {}
        if itr is None:
            # used last itr available, should be rare
            res['itr'] = '-2'  # TODO: could replace with last observed itr, too much trouble for now
        else:
            if 'best' in itr:
                if itr == 'best':
                    best_itr = get_best_itr(variant, use_min=use_min)
                    res['itr'] = best_itr
                else:
                    # assuming form 'best<[X]'
                    assert '<' in itr
                    best_itr_max = int(itr.split('<')[1])
                    best_itr = get_best_itr(variant, max=best_itr_max, use_min=use_min)
                    res['itr'] = best_itr
            else:
                res['itr'] = itr
        simulate = simulate[288:288*(n_days+1)]
        if night_only:
            idx = simulate.index
            idx_bool = idx.time < datetime.time(hour=6)
            simulate = simulate[idx_bool]
        if meal_only:
            idx_bool = simulate['CHO'].rolling(window=48).max().fillna(0) > 0
            simulate = simulate[idx_bool]
        sim_res = get_res_dict(simulate)
        res.update(sim_res)
        pairs = [('return', 'Test Returns Mean'), ('glen', 'GLen'),
                 ('progress_euglycemic', 'Euglycemic'),
                 ('progress_hypo', 'Hypoglycemic'),
                 ('progress_hyper', 'Hyperglycemic'), ]
        for pair in pairs:
            label_name, progress_name = pair
            if progress_name in progress:
                res['{}_final'.format(label_name)] = progress[progress_name].values[-1]
                try:
                    res['{}_max'.format(label_name)] = progress[progress_name].max()
                    res['{}_max_ind'.format(label_name)] = progress[progress_name].idxmax()
                except:
                    raise ValueError('{}, {}'.format(fl, progress_name))
    except:
        print(res)
        # raise
        return {}
    return res


def rlkit_load_and_summarize(directory, fl, itr, run_if_absent, replace_if_present, save_q, cpu_only, gpu_remap,
                             new_eval, bgp_dir, eval_seed, n_eval_runs, night_only, meal_only,
                             n_eval_start, n_days, suffix, use_min):
    if n_eval_runs is None:
        return _rlkit_load_and_summarize(directory, fl, itr, run_if_absent, replace_if_present, save_q, cpu_only, gpu_remap,
                             new_eval, bgp_dir, eval_seed, None, night_only, meal_only, n_days, suffix, use_min)
    else:
        res_arr = []
        for i in range(n_eval_start, n_eval_start+n_eval_runs):
            res = _rlkit_load_and_summarize(directory, fl, itr, run_if_absent, replace_if_present, save_q, cpu_only,
                                            gpu_remap, new_eval, bgp_dir, eval_seed, i, night_only, meal_only,
                                            n_days, suffix, use_min)
            res_arr.append(res)
        return res_arr
