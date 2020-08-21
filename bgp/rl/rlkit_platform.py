import numpy as np
import joblib
import pandas as pd
import os
from tqdm import tqdm
import torch
from bgp.rlkit.launchers.launcher_util import setup_logger
from bgp.rlkit.torch.sac.sac import SoftActorCritic
from bgp.rlkit.torch.sac.policies import (CNNTanhGaussianPolicy, GRUTanhGaussianPolicy,
                                          TanhGaussianPolicy, FancyCNNTanhGaussianPolicy)
from bgp.rlkit.torch.networks import SimpleCNNQ, SimpleGRUQ, FancyCNNQ, FlattenMlp
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
from bgp.rl import reward_functions


def reward_name_to_function(reward_name):
    if reward_name == 'risk_diff':
        reward_fun = reward_functions.risk_diff
    elif reward_name == 'risk_diff_bg':
        reward_fun = reward_functions.risk_diff_bg
    elif reward_name == 'risk':
        reward_fun = reward_functions.reward_risk
    elif reward_name == 'risk_bg':
        reward_fun = reward_functions.risk_bg
    elif reward_name == 'magni_bg':
        reward_fun = reward_functions.magni_reward
    elif reward_name == 'cameron_bg':
        reward_fun = reward_functions.cameron_reward
    elif reward_name == 'eps_risk':
        reward_fun = reward_functions.epsilon_risk
    elif reward_name == 'target_bg':
        reward_fun = reward_functions.reward_target
    elif reward_name == 'cgm_high':
        reward_fun = reward_functions.reward_cgm_high
    elif reward_name == 'bg_high':
        reward_fun = reward_functions.reward_bg_high
    elif reward_name == 'cgm_low':
        reward_fun = reward_functions.reward_cgm_low
    else:
        raise ValueError('{} not a proper reward_name'.format(reward_name))
    return reward_fun


def get_best_itr(variant, max=None, seed=None, use_min=True):
    # Note: currently always loading up run that minimizes risk, could optimize for other things
    try:
        if seed is None:
            progress = pd.read_csv('{}/progress.csv'.format(variant['log_dir']))
        else:
            progress = pd.read_csv('{}/progress_seed{}.csv'.format(variant['log_dir'], seed))
    except pd.errors.EmptyDataError as e:
        return -1
    files = os.listdir(variant['log_dir'])
    files = list(filter(lambda x: 'itr_' in x, files))
    eval_itrs = list(filter(lambda x: 'eval' in x, files))

    if len(eval_itrs) > 0:
        # new eval strat, should have eval_itr for each iteration
        cached_itrs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[2]), eval_itrs)))
    else:
        # old method, should only have itr_x.pkl
        cached_itrs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), files)))
    cached_itrs = list(filter(lambda x: x >= 0 and x <= list(progress.index)[-1], cached_itrs))
    if max is not None:
        cached_itrs = list(filter(lambda x: x < max, cached_itrs))
    pcache = progress.iloc[np.array(cached_itrs)]
    # first, restrict search to entries that maximize length alive
    pcache_filter = pcache[pcache['GLen'] == pcache['GLen'].max()]
    # second, try to enforce some safety constraints
    if 'MinBG' in pcache_filter and use_min:
        print('Using Min {}'.format(use_min))
        if use_min is True:
            min_lvl = 30
        else:
            min_lvl = use_min
        try:
            if pcache_filter['MinBG'].max() < min_lvl:
                min_lvl = pcache_filter['MinBG'].max() * 0.9
        except:
            print(min_lvl, type(min_lvl), pcache_filter['MinBG'].max())
        pcache_filter = pcache_filter[pcache_filter['MinBG'] >= min_lvl]
    # third, choose most performant policy
    best_itr = pcache_filter['Test Rewards Mean'].idxmax()
    print('Best Itr: {}/{}'.format(best_itr, len(progress['Test Rewards Mean'])))
    return best_itr


def simulate_policy(variant, itr=None, save_q=False, cpu_only=False, gpu_remap=None, new_eval=True,
                    progress_seed=None, sim_seed_submod=0, lab=False, use_min=True):
    if new_eval:
        prefix = 'eval_'
    else:
        prefix = ''
    if itr is None:
        data = joblib.load('{}/{}params.pkl'.format(variant['log_dir'], prefix))
    elif 'best' in itr:
        if itr == 'best':
            best_itr = get_best_itr(variant, seed=progress_seed, use_min=use_min)
            joblib.dump(best_itr, '{}/best_itr={}.pkl'.format(variant['log_dir'], best_itr))
            if best_itr == -1:
                # Nothing has actually run
                return -1
            data = joblib.load('{}/{}itr_{}.pkl'.format(variant['log_dir'], prefix, best_itr))
        else:
            # assuming form 'best<[X]'
            assert '<' in itr
            print('in best< form')
            best_itr_max = int(itr.split('<')[1])
            best_itr = get_best_itr(variant, max=best_itr_max, use_min=use_min)
            joblib.dump(best_itr, '{}/best_itr<{}={}.pkl'.format(variant['log_dir'], best_itr_max, best_itr))
            if best_itr == -1:
                # Nothing has actually run
                return -1
            data = joblib.load('{}/{}itr_{}.pkl'.format(variant['log_dir'], prefix, best_itr))
    else:
        print('Loading {}'.format(itr))
        data = joblib.load('{}/{}itr_{}.pkl'.format(variant['log_dir'], prefix, itr))
    if new_eval:
        mdl = data
    else:
        mdl = data['eval_policy']
    reward_fun = reward_name_to_function(variant['reward_fun'])
    if cpu_only:
        run_device = 'cpu'
    elif gpu_remap is not None and variant['device'] in gpu_remap:
        run_device = gpu_remap[variant['device']]
    else:
        run_device = variant['device']
    print(run_device)
    if variant['model_type'] == 'sac':
        mdl.stochastic_policy.device = run_device
        mdl.stochastic_policy = mdl.stochastic_policy.to(run_device)
        mdl.stochastic_policy.eval()
        if mdl.stochastic_policy.features is not None:
            mdl.stochastic_policy.features.device = run_device
            mdl.stochastic_policy.features = mdl.stochastic_policy.features.to(run_device)
            mdl.stochastic_policy.features.eval()
        env = bgp_env.DeepSACT1DEnv(reward_fun=reward_fun,
                                    patient_name=variant['patient_name'],
                                    seeds={'numpy': variant['base_seed'] + variant['sim_seed_mod'] + sim_seed_submod,
                                           'sensor': variant['base_seed'] + variant['sim_seed_mod'] + sim_seed_submod,
                                           'scenario': variant['base_seed'] + variant['sim_seed_mod'] + sim_seed_submod},
                                    reset_lim=variant['reset_lim'], time=variant['include_time'],
                                    meal=variant['include_meal'], bw_meals=variant['bw_meals'], load=variant['load'],
                                    use_pid_load=variant['use_pid_load'], hist_init=variant['hist_init'],
                                    gt=variant['use_ground_truth'], n_hours=variant['n_hours'], norm=variant['norm'],
                                    time_std=variant['time_std'], use_old_patient_env=variant['use_old_patient_env'],
                                    action_cap=variant['action_cap'], action_bias=variant['action_bias'],
                                    action_scale=variant['action_scale'], basal_scaling=variant['basal_scaling'],
                                    meal_announce=variant['meal_announce'],
                                    residual_basal=variant['residual_basal'], residual_bolus=variant['residual_bolus'],
                                    residual_PID=variant['residual_PID'],
                                    fake_gt=variant['fake_gt'], fake_real=variant['fake_real'],
                                    suppress_carbs=variant['suppress_carbs'], limited_gt=variant['limited_gt'],
                                    termination_penalty=variant['termination_penalty'], weekly=variant['weekly'],
                                    update_seed_on_reset=variant['update_seed_on_reset'],
                                    deterministic_meal_size=variant['deterministic_meal_size'],
                                    deterministic_meal_time=variant['deterministic_meal_time'],
                                    deterministic_meal_occurrence=variant['deterministic_meal_occurrence'],
                                    harrison_benedict=variant['harrison_benedict_sched'],
                                    restricted_carb=variant['restricted_sched'], meal_duration=variant['meal_duration'],
                                    rolling_insulin_lim=variant['rolling_insulin_lim'], universal=variant['universal'],
                                    reward_bias=variant['reward_bias'], carb_error_std=variant['carb_error_std'],
                                    carb_miss_prob=variant['carb_miss_prob'], source_dir=variant['source_dir'])
    else:
        raise ValueError('No proper model type given: {}'.format(variant['model_type']))
    if lab:
        return mdl, env
    q_save = []
    for _ in tqdm(range(variant['n_sim_days']*288)):
        if save_q:
            if variant['rnn']:  # TODO: not sure about this
                input = torch.tensor(env.get_state(variant['norm']), device=run_device).float()[:, None]
            else:
                input = torch.tensor(env.get_state(variant['norm']), device=run_device).float()[None, :]
            q = mdl.qf(input).detach().cpu().numpy()
            q_save.append(q)
        try:
            a, _ = mdl.get_action(env.get_state(variant['norm']))
        except:
            mdl.stochastic_policy = mdl.stochastic_policy.to(run_device)
            raise ValueError(mdl.stochastic_policy.features.device, run_device,
                             mdl.stochastic_policy.last_fc.weight.device,
                             mdl.stochastic_policy.last_fc_log_std.weight.device)
        if variant['model_type'] == 'sac':
            a = a.item()
        env.step(action=a)
    if save_q:
        q_save = np.concatenate(q_save)
        adv_save = q_save - np.max(q_save, axis=1)[:, None]
        return env.env.show_history(), q_save, adv_save
    else:

        return env.env.show_history()

def run_em_sac(variant):
    # why no logger?
    if variant['independent_init'] is not None:
        torch.manual_seed(variant['independent_init'])
    else:
        torch.manual_seed(variant['base_seed'])
    setup_logger(variant['run_name'], variant=variant, log_dir=variant['log_dir'],
                 snapshot_mode='gap_and_last', snapshot_gap=variant['snapshot_gap'])
    # why no eval environment?

    reward_fun = reward_name_to_function(variant['reward_fun'])
    env = bgp_env.DeepSACT1DEnv(reward_fun=reward_fun,
                                patient_name=variant['patient_name'],
                                seeds={'numpy': variant['base_seed'],
                                       'sensor': variant['base_seed'],
                                       'scenario': variant['base_seed']},
                                reset_lim=variant['reset_lim'], time=variant['include_time'],
                                meal=variant['include_meal'], bw_meals=variant['bw_meals'],
                                load=variant['load'], use_pid_load=variant['use_pid_load'],
                                hist_init=variant['hist_init'],
                                gt=variant['use_ground_truth'], n_hours=variant['n_hours'],
                                norm=variant['norm'], time_std=variant['time_std'],
                                use_old_patient_env=variant['use_old_patient_env'], action_cap=variant['action_cap'],
                                action_bias=variant['action_bias'], action_scale=variant['action_scale'],
                                basal_scaling=variant['basal_scaling'],
                                meal_announce=variant['meal_announce'], residual_basal=variant['residual_basal'],
                                residual_bolus=variant['residual_bolus'], residual_PID=variant['residual_PID'],
                                fake_gt=variant['fake_gt'], fake_real=variant['fake_real'],
                                suppress_carbs=variant['suppress_carbs'], limited_gt=variant['limited_gt'],
                                termination_penalty=variant['termination_penalty'], weekly=variant['weekly'],
                                update_seed_on_reset=variant['update_seed_on_reset'],
                                deterministic_meal_size=variant['deterministic_meal_size'],
                                deterministic_meal_time=variant['deterministic_meal_time'],
                                deterministic_meal_occurrence=variant['deterministic_meal_occurrence'],
                                harrison_benedict=variant['harrison_benedict_sched'],
                                restricted_carb=variant['restricted_sched'], meal_duration=variant['meal_duration'],
                                rolling_insulin_lim=variant['rolling_insulin_lim'], universal=variant['universal'],
                                reward_bias=variant['reward_bias'], carb_error_std=variant['carb_error_std'],
                                carb_miss_prob=variant['carb_miss_prob'], source_dir=variant['source_dir'])

    obs_dim = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))

    if variant['use_ground_truth']:
        obs_dim = int(np.prod(obs_dim))
        qf = FlattenMlp(
            hidden_sizes=[variant['net_size'], variant['net_size']],
            input_size=obs_dim + action_dim,
            output_size=1,
            layer_norm=variant['layernorm'],
            device=variant['device']
        )
        vf = FlattenMlp(
            hidden_sizes=[variant['net_size'], variant['net_size']],
            input_size=obs_dim,
            output_size=1,
            layer_norm=variant['layernorm'],
            device=variant['device']
        )
        policy = TanhGaussianPolicy(
            hidden_sizes=[variant['net_size'], variant['net_size']],
            obs_dim=obs_dim,
            action_dim=action_dim,
            layer_norm=variant['layernorm'],
            device=variant['device']
        )
    elif variant['fancy']:
        qf = FancyCNNQ(
            input_size=[obs_dim[0] + 1, obs_dim[1]],
            output_size=1,
            device=variant['device']
        )
        vf = FancyCNNQ(
            input_size=obs_dim,
            output_size=1,
            device=variant['device']
        )
        policy = FancyCNNTanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=variant['device']
        )
    elif variant['rnn']:
        qf = SimpleGRUQ(
            input_size=[obs_dim[0]+1, obs_dim[1]],
            output_size=1,
            device=variant['device'],
            hidden_size=variant['rnn_size'],
            num_layers=variant['rnn_layers'],
            dilation=variant['dilation']
        )
        vf = SimpleGRUQ(
            input_size=obs_dim,
            output_size=1,
            device=variant['device'],
            hidden_size=variant['rnn_size'],
            num_layers=variant['rnn_layers'],
            dilation=variant['dilation']
        )
        policy = GRUTanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=variant['device'],
            hidden_size=variant['rnn_size'],
            num_layers=variant['rnn_layers'],
            dilation=variant['dilation']
        )
    else:
        qf = SimpleCNNQ(
            input_size=[obs_dim[0]+1, obs_dim[1]],
            output_size=1,
            device=variant['device']
        )
        vf = SimpleCNNQ(
            input_size=obs_dim,
            output_size=1,
            device=variant['device']
        )
        policy = CNNTanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=variant['device']
        )
    replay_buffer = None
    torch.manual_seed(variant['base_seed'])
    if variant['warm_start'] is not None:
        # assume that this gives the directory name to transfer from, if their name is available choose that, else
        # choose their type (child, adolescent, adult)
        files = os.listdir('/data/dir/{}'.format(variant['warm_start']))
        if '{}.pkl'.format(variant['patient_name']) in files:
            param_dict = joblib.load('/data/dir/{}/{}.pkl'.format(variant['warm_start'],
                                                                                variant['patient_name']))
        elif '{}.pkl'.format(variant['patient_name'].split('#')[0]) in files:
            param_dict = joblib.load('/data/dir/{}/{}.pkl'.format(variant['warm_start'],
                                                                                variant['patient_name'].split('#')[0]))
        elif '{}#001.pkl'.format(variant['patient_name'].split('#')[0]) in files:
            param_dict = joblib.load('/data/dir/{}/{}#001.pkl'.format(variant['warm_start'],
                                                                                    variant['patient_name'].split('#')[0]))
        else:
            print('falling back on default')
            param_dict = joblib.load('/data/dir/{}/full.pkl'.format(variant['warm_start']))
        qf.load_state_dict(param_dict['qf'])
        vf.load_state_dict(param_dict['vf'])
        policy.load_state_dict(param_dict['policy'])
        if 'replay_buffer' in param_dict:
            replay_buffer = param_dict['replay_buffer']
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        loss_criterion=variant['loss_function'],
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    algorithm.to(variant['device'])
    algorithm.train()
    print('final simulations')
    for submod in range(variant['num_eval_runs']):
        hist = simulate_policy(variant, itr='best', sim_seed_submod=submod)
        if submod == 0:
            # some backwards compatibility
            joblib.dump(hist, '{}/simulate_best.pkl'.format(variant['log_dir']))
        joblib.dump(hist, '{}/simulate_best_{}.pkl'.format(variant['log_dir'], submod))


def finish_sac(variant):
    if variant['num_eval_runs'] == 1:
        variant['num_eval_runs'] = 100
    if variant['finish_mod'] is not None:
        suffix = '_{}'.format(variant['finish_mod'])
    else:
        suffix = ''
    for submod in range(variant['num_eval_runs']):
        print('Using {}'.format(variant['finish_itr']))
        hist = simulate_policy(variant, itr=variant['finish_itr'], sim_seed_submod=submod, use_min=variant['use_min'])
        if submod == 0:
            # some backwards compatibility
            joblib.dump(hist, '{}/simulate_best{}.pkl'.format(variant['log_dir'], suffix))
        joblib.dump(hist, '{}/simulate_best_{}{}.pkl'.format(variant['log_dir'], submod, suffix))
