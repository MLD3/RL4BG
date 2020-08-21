from collections import namedtuple

ShortState = namedtuple('ShortState', ['insulin', 'bg'])
CLStateBins = namedtuple('CLStateBins', ['bg_bins', 'insulin_bins'])
ShortAction = namedtuple('ShortAction', ['basal', 'bolus'])

LongState = namedtuple('LongState', ['hypo', 'hyper'])
LongAction = namedtuple('LongAction', ['basal_rate_incr', 'correction_factor_incr', 'carbohydrate_ratio_incr'])
OLStateBins = namedtuple('OLStateBins', ['hypo_bins', 'hyper_bins'])
OLActionBins = namedtuple('OLActionBins', ['basal_rate_bins', 'carbohydrate_ratio_bins'])

Action = namedtuple('Action', ['basal', 'bolus'])
ResetLim = namedtuple('ResetLim', ['lower_lim', 'upper_lim'])
Seed = namedtuple('Seed', ['sensor_seed', 'scenario_seed', 'numpy_seed'])

TabularQ_State = namedtuple('tabularq_state', ['insulin', 'bg'])
RLConfig = namedtuple('RLConfig', ['n_days_train', 'n_days_eval', 'epsilon', 'burn_in', 'target_bg', 'discount_factor',
                                   'name', 'reset_lower_lim', 'reset_upper_lim', 'sensor_seed', 'scenario_seed',
                                   'closed_loop_action_bins', 'insulin_bins', 'bg_bins', 'lr', 'save_name', 'numpy_seed'])

SimConfig = namedtuple('SimConfig', ['name', 'sensor_seed', 'scenario_seed'])
EnvConfig = namedtuple('EnvConfig', ['sim_config', 'insulin_bins', 'bg_bins', 'closed_loop_action_bins', 'long_action_bins',
                                     'hyper_bins', 'hypo_bins', 'reset_lower_lim', 'reset_upper_lim'])

NewEnvConfig = namedtuple('NewEnvConfig', ['name', 'seeds', 'state_bins', 'action_bins', 'reset_lim', 'reward_fun',
                                           'sample_time', 'meal_mult'])

# TODO: could separate model-specific features in training configs
ModelConfig = namedtuple('ModelConfig', ['model_type', 'input_size', 'action_size', 'tabular_action', 'init',
                                         'basis_bins', 'device'])
TrainConfig = namedtuple('TrainConfig', ['save_name', 'numpy_seed', 'n_days_train', 'n_days_eval', 'epsilon',
                                         'burn_in', 'lr', 'discount_factor', 'std', 'n_boot', 'normalize_reward',
                                         'require_full', 'batch_size', 'memory_capacity', 'update_target_every',
                                         'save_model_every', 'eval_same', 'n_days_init', 'pid_param'])
# Old, phased out
#TrainConfig = namedtuple('TrainConfig', ['save_name', 'numpy_seed', 'n_days_train', 'n_days_eval', 'epsilon',
#                                         'burn_in', 'lr', 'discount_factor', 'std', 'n_boot', 'normalize_reward',
#                                         'require_full', 'update_every'])
TotalConfig = namedtuple('TotalConfig', ['env_config', 'model_config', 'train_config'])

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
