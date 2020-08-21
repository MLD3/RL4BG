from bgp.simglucose.simulation.env import T1DSimEnv
from bgp.simglucose.patient.t1dpatient import T1DPatientNew
from bgp.simglucose.sensor.cgm import CGMSensor
from bgp.simglucose.actuator.pump import InsulinPump
from bgp.simglucose.simulation.scenario_gen import (RandomBalancedScenario, SemiRandomBalancedScenario,
                                                    CustomBalancedScenario)
from bgp.simglucose.controller.base import Action
from bgp.simglucose.analysis.risk import magni_risk_index
from bgp.rl.helpers import Seed
from bgp.rl import pid

import pandas as pd
import numpy as np
import joblib
import copy
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class DeepSACT1DEnv(gym.Env):
    '''
    A gym environment supporting SAC learning. Uses PID control for initialization
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_fun, patient_name=None, seeds=None,
                 reset_lim=None, time=False, meal=False, gt=False, load=False,
                 bw_meals=False, n_hours=24, time_std=None, norm=False, use_old_patient_env=False, action_cap=0.1,
                 action_bias=0, action_scale=1, basal_scaling=216.0, meal_announce=None,
                 residual_basal=False, residual_bolus=False, residual_PID=False, fake_gt=False, fake_real=False,
                 suppress_carbs=False, limited_gt=False, termination_penalty=None, weekly=False,
                 use_model=False, model=None, model_device='cpu', update_seed_on_reset=False,
                 deterministic_meal_size=False, deterministic_meal_time=False, deterministic_meal_occurrence=False,
                 use_pid_load=False, hist_init=False, start_date=None, use_custom_meal=False, custom_meal_num=3,
                 custom_meal_size=1, starting_glucose=None,
                 harrison_benedict=False, restricted_carb=False, meal_duration=1, rolling_insulin_lim=None,
                 universal=False, unrealistic=False, reward_bias=0, carb_error_std=0, carb_miss_prob=0, source_dir=None,
                 **kwargs):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.source_dir = source_dir
        self.patient_para_file = '{}/bgp/simglucose/params/vpatient_params.csv'.format(self.source_dir)
        self.control_quest = '{}/bgp/simglucose/params/Quest2.csv'.format(self.source_dir)
        self.pid_para_file = '{}/bgp/simglucose/params/pid_params.csv'.format(self.source_dir)
        self.pid_env_path = '{}/bgp/simglucose/params'.format(self.source_dir)
        self.sensor_para_file = '{}/bgp/simglucose/params/sensor_params.csv'.format(self.source_dir)
        self.insulin_pump_para_file = '{}/bgp/simglucose/params/pump_params.csv'.format(self.source_dir)
        # reserving half of pop for testing
        self.universe = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 6)] +
                         ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 6)] +
                         ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 6)])
        self.universal = universal
        if seeds is None:
            seed_list = self._seed()
            seeds = Seed(numpy_seed=seed_list[0], sensor_seed=seed_list[1], scenario_seed=seed_list[2])
        if patient_name is None:
            if self.universal:
                patient_name = np.random.choice(self.universe)
            else:
                patient_name = 'adolescent#001'
        np.random.seed(seeds['numpy'])
        self.seeds = seeds
        self.sample_time = 5
        self.day = int(1440 / self.sample_time)
        self.state_hist = int((n_hours * 60) / self.sample_time)
        self.time = time
        self.meal = meal
        self.norm = norm
        self.gt = gt
        self.reward_fun = reward_fun
        self.reward_bias = reward_bias
        self.action_cap = action_cap
        self.action_bias = action_bias
        self.action_scale = action_scale
        self.basal_scaling = basal_scaling
        self.meal_announce = meal_announce
        self.meal_duration = meal_duration
        self.deterministic_meal_size = deterministic_meal_size
        self.deterministic_meal_time = deterministic_meal_time
        self.deterministic_meal_occurrence = deterministic_meal_occurrence
        self.residual_basal = residual_basal
        self.residual_bolus = residual_bolus
        self.carb_miss_prob = carb_miss_prob
        self.carb_error_std = carb_error_std
        self.residual_PID = residual_PID
        self.use_pid_load = use_pid_load
        self.fake_gt = fake_gt
        self.fake_real = fake_real
        self.suppress_carbs = suppress_carbs
        self.limited_gt = limited_gt
        self.termination_penalty = termination_penalty
        self.target = 140
        self.low_lim = 140  # Matching BB controller
        self.cooldown = 180
        self.last_cf = self.cooldown + 1
        self.start_date = start_date
        self.rolling_insulin_lim = rolling_insulin_lim
        self.rolling = []
        if self.start_date is None:
            start_time = datetime(2018, 1, 1, 0, 0, 0)
        else:
            start_time = datetime(self.start_date.year, self.start_date.month, self.start_date.day, 0, 0, 0)
        assert bw_meals  # otherwise code wouldn't make sense
        if reset_lim is None:
            self.reset_lim = {'lower_lim': 10, 'upper_lim': 1000}
        else:
            self.reset_lim = reset_lim
        self.load = load
        self.hist_init = hist_init
        self.env = None
        self.use_old_patient_env = use_old_patient_env
        self.model = model
        self.model_device = model_device
        self.use_model = use_model
        self.harrison_benedict = harrison_benedict
        self.restricted_carb = restricted_carb
        self.unrealistic = unrealistic
        self.start_time = start_time
        self.time_std = time_std
        self.weekly = weekly
        self.update_seed_on_reset = update_seed_on_reset
        self.use_custom_meal = use_custom_meal
        self.custom_meal_num = custom_meal_num
        self.custom_meal_size = custom_meal_size
        self.set_patient_dependent_values(patient_name)
        self.env.scenario.day = 0

    def pid_load(self, n_days):
        for i in range(n_days*self.day):
            b_val = self.pid.step(self.env.CGM_hist[-1])
            act = Action(basal=0, bolus=b_val)
            _ = self.env.step(action=act, reward_fun=self.reward_fun, cho=None)

    def step(self, action):
        return self._step(action, cho=None)

    def translate(self, action):
        if self.action_scale == 'basal':
            # 288 samples per day, bolus insulin should be 75% of insulin dose
            # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
            # https://care.diabetesjournals.org/content/34/5/1089
            action = (action + self.action_bias) * ((self.ideal_basal * self.basal_scaling) / (1 + self.action_bias))
        else:
            action = (action + self.action_bias) * self.action_scale
        return max(0, action)

    def _step(self, action, cho=None, use_action_scale=True):
        # cho controls if carbs are eaten, else taken from meal policy
        if type(action) is np.ndarray:
            action = action.item()
        if use_action_scale:
            if self.action_scale == 'basal':
                # 288 samples per day, bolus insulin should be 75% of insulin dose
                # split over 4 meals with 5 minute sampling rate, max unscaled value is 1+action_bias
                # https://care.diabetesjournals.org/content/34/5/1089
                action = (action + self.action_bias) * ((self.ideal_basal * self.basal_scaling)/(1+self.action_bias))
            else:
                action = (action + self.action_bias) * self.action_scale
        if self.residual_basal:
            action += self.ideal_basal
        if self.residual_bolus:
            ma = self.announce_meal(5)
            carbs = ma[0]
            if np.random.uniform() < self.carb_miss_prob:
                carbs = 0
            error = np.random.normal(0, self.carb_error_std)
            carbs = carbs + carbs * error
            glucose = self.env.CGM_hist[-1]
            if carbs > 0:
                carb_correct = carbs / self.CR
                hyper_correct = (glucose > self.target) * (glucose - self.target) / self.CF
                hypo_correct = (glucose < self.low_lim) * (self.low_lim - glucose) / self.CF
                bolus = 0
                if self.last_cf > self.cooldown:
                    bolus += hyper_correct - hypo_correct
                bolus += carb_correct
                action += bolus / 5.
                self.last_cf = 0
            self.last_cf += 5
        if self.residual_PID:
            action += self.pid.step(self.env.CGM_hist[-1])
        if self.action_cap is not None:
            action = min(self.action_cap, action)
        if self.rolling_insulin_lim is not None:
            if np.sum(self.rolling + [action]) > self.rolling_insulin_lim:
                action = max(0, action - (np.sum(self.rolling + [action]) - self.rolling_insulin_lim))
            self.rolling.append(action)
            if len(self.rolling) > 12:
                self.rolling = self.rolling[1:]
        act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun, cho=cho)
        state = self.get_state(self.norm)
        done = self.is_done()
        if done and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        reward = reward + self.reward_bias
        return state, reward, done, info

    def announce_meal(self, meal_announce=None):
        t = self.env.time.hour * 60 + self.env.time.minute  # Assuming 5 minute sampling rate
        for i, m_t in enumerate(self.env.scenario.scenario['meal']['time']):
            # round up to nearest 5
            if m_t % 5 != 0:
                m_tr = m_t - (m_t % 5) + 5
            else:
                m_tr = m_t
            if meal_announce is None:
                ma = self.meal_announce
            else:
                ma = meal_announce
            if t < m_tr <= t + ma:
                return self.env.scenario.scenario['meal']['amount'][i], m_tr - t
        return 0, 0

    def calculate_iob(self):
        ins = self.env.insulin_hist
        return np.dot(np.flip(self.iob, axis=0)[-len(ins):], ins[-len(self.iob):])

    def get_state(self, normalize=False):
        bg = self.env.CGM_hist[-self.state_hist:]
        insulin = self.env.insulin_hist[-self.state_hist:]
        if normalize:
            bg = np.array(bg)/400.
            insulin = np.array(insulin) * 10
        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate((np.full(self.state_hist - len(insulin), -1), insulin))
        return_arr = [bg, insulin]
        if self.time:
            time_dt = self.env.time_hist[-self.state_hist:]
            time = np.array([(t.minute + 60 * t.hour) / self.sample_time for t in time_dt])
            sin_time = np.sin(time * 2 * np.pi / self.day)
            cos_time = np.cos(time * 2 * np.pi / self.day)
            if normalize:
                pass  # already normalized
            if len(sin_time) < self.state_hist:
                sin_time = np.concatenate((np.full(self.state_hist - len(sin_time), -1), sin_time))
            if len(cos_time) < self.state_hist:
                cos_time = np.concatenate((np.full(self.state_hist - len(cos_time), -1), cos_time))
            return_arr.append(sin_time)
            return_arr.append(cos_time)
            if self.weekly:
                # binary flag signalling weekend
                if self.env.scenario.day == 5 or self.env.scenario.day == 6:
                    return_arr.append(np.full(self.state_hist, 1))
                else:
                    return_arr.append(np.full(self.state_hist, 0))
        if self.meal:
            cho = self.env.CHO_hist[-self.state_hist:]
            if normalize:
                cho = np.array(cho)/20.
            if len(cho) < self.state_hist:
                cho = np.concatenate((np.full(self.state_hist - len(cho), -1), cho))
            return_arr.append(cho)
        if self.meal_announce is not None:
            meal_val, meal_time = self.announce_meal()
            future_cho = np.full(self.state_hist, meal_val)
            return_arr.append(future_cho)
            future_time = np.full(self.state_hist, meal_time)
            return_arr.append(future_time)
        if self.fake_real:
            state = self.env.patient.state
            return np.stack([state for _ in range(self.state_hist)]).T.flatten()
        if self.gt:
            if self.fake_gt:
                iob = self.calculate_iob()
                cgm = self.env.CGM_hist[-1]
                if normalize:
                    state = np.array([cgm/400., iob*10])
                else:
                    state = np.array([cgm, iob])
            else:
                state = self.env.patient.state
            if self.meal_announce is not None:
                meal_val, meal_time = self.announce_meal()
                state = np.concatenate((state, np.array([meal_val, meal_time])))
            if normalize:
                # just the average of 2 days of adult#001, these values are patient-specific
                norm_arr = np.array([4.86688301e+03, 4.95825609e+03, 2.52219425e+03, 2.73376341e+02,
                                     1.56207049e+02, 9.72051746e+00, 7.65293763e+01, 1.76808549e+02,
                                     1.76634852e+02, 5.66410518e+00, 1.28448645e+02, 2.49195394e+02,
                                     2.73250649e+02, 7.70883882e+00, 1.63778163e+00])
                if self.meal_announce is not None:
                    state = state/norm_arr
                else:
                    state = state/norm_arr[:-2]
            if self.suppress_carbs:
                state[:3] = 0.
            if self.limited_gt:
                state = np.array([state[3], self.calculate_iob()])
            return state
        return np.stack(return_arr).flatten()

    def avg_risk(self):
        return np.mean(self.env.risk_hist[max(self.state_hist, 288):])

    def avg_magni_risk(self):
        return np.mean(self.env.magni_risk_hist[max(self.state_hist, 288):])

    def glycemic_report(self):
        bg = np.array(self.env.BG_hist[max(self.state_hist, 288):])
        ins = np.array(self.env.insulin_hist[max(self.state_hist, 288):])
        hypo = (bg < 70).sum()/len(bg)
        hyper = (bg > 180).sum()/len(bg)
        euglycemic = 1 - (hypo+hyper)
        return bg, euglycemic, hypo, hyper, ins

    def is_done(self):
        return self.env.BG_hist[-1] < self.reset_lim['lower_lim'] or self.env.BG_hist[-1] > self.reset_lim['upper_lim']

    def increment_seed(self, incr=1):
        self.seeds['numpy'] += incr
        self.seeds['scenario'] += incr
        self.seeds['sensor'] += incr

    def reset(self):
        return self._reset()

    def set_patient_dependent_values(self, patient_name):
        self.patient_name = patient_name
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split('#')[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))['BW'].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))['u2ss'].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        if self.rolling_insulin_lim is not None:
            self.rolling_insulin_lim = ((self.rolling_insulin_lim * self.bw) / self.CR * self.rolling_insulin_lim) / 5
        else:
            self.rolling_insulin_lim = None
        iob_all = joblib.load('{}/iob.pkl'.format(self.pid_env_path))
        self.iob = iob_all[self.patient_name]
        pid_df = pd.read_csv(self.pid_para_file)
        if patient_name not in pid_df.name.values:
            raise ValueError('{} not in PID csv'.format(patient_name))
        pid_params = pid_df.loc[pid_df.name == patient_name].squeeze()
        self.pid = pid.PID(setpoint=pid_params.setpoint,
                           kp=pid_params.kp, ki=pid_params.ki, kd=pid_params.kd)
        patient = T1DPatientNew.withName(patient_name, self.patient_para_file)
        sensor = CGMSensor.withName('Dexcom', self.sensor_para_file, seed=self.seeds['sensor'])
        if self.time_std is None:
            scenario = RandomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                              kind=self.kind, restricted=self.restricted_carb,
                                              harrison_benedict=self.harrison_benedict, unrealistic=self.unrealistic,
                                              deterministic_meal_size=self.deterministic_meal_size,
                                              deterministic_meal_time=self.deterministic_meal_time,
                                              deterministic_meal_occurrence=self.deterministic_meal_occurrence,
                                              meal_duration=self.meal_duration)
        elif self.use_custom_meal:
            scenario = CustomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                              num_meals=self.custom_meal_num, size_mult=self.custom_meal_size)
        else:
            scenario = SemiRandomBalancedScenario(bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
                                                  time_std_multiplier=self.time_std, kind=self.kind,
                                                  harrison_benedict=self.harrison_benedict,
                                                  meal_duration=self.meal_duration)
        pump = InsulinPump.withName('Insulet', self.insulin_pump_para_file)
        self.env = T1DSimEnv(patient=patient,
                             sensor=sensor,
                             pump=pump,
                             scenario=scenario,
                             sample_time=self.sample_time, source_dir=self.source_dir)
        if self.hist_init:
            self.env_init_dict = joblib.load("{}/{}_data.pkl".format(self.pid_env_path, self.patient_name))
            self.env_init_dict['magni_risk_hist'] = []
            for bg in self.env_init_dict['bg_hist']:
                self.env_init_dict['magni_risk_hist'].append(magni_risk_index([bg]))
            self._hist_init()

    def _reset(self):
        if self.update_seed_on_reset:
            self.increment_seed()
        if self.use_model:
            if self.load:
                self.env = joblib.load("{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name))
                self.env.model = self.model
                self.env.model_device = self.model_device
                self.env.norm_params = self.norm_params
                self.env.state = self.env.patient.state
                self.env.scenario.kind = self.kind
            else:
                self.env.reset()
        else:
            if self.load:
                if self.use_old_patient_env:
                    self.env = joblib.load("{}/{}_env.pkl".format(self.pid_env_path, self.patient_name))
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                else:
                    self.env = joblib.load("{}/{}_fenv.pkl".format(self.pid_env_path, self.patient_name))
                    self.env.model = None
                    self.env.scenario.kind = self.kind
                if self.time_std is not None:
                    self.env.scenario = SemiRandomBalancedScenario(bw=self.bw, start_time=self.start_time,
                                                                   seed=self.seeds['scenario'],
                                                                   time_std_multiplier=self.time_std, kind=self.kind,
                                                                   harrison_benedict=self.harrison_benedict,
                                                                   meal_duration=self.meal_duration)
                self.env.sensor.seed = self.seeds['sensor']
                self.env.scenario.seed = self.seeds['scenario']
                self.env.scenario.day = 0
                self.env.scenario.weekly = self.weekly
                self.env.scenario.kind = self.kind
            else:
                if self.universal:
                    patient_name = np.random.choice(self.universe)
                    self.set_patient_dependent_values(patient_name)
                self.env.sensor.seed = self.seeds['sensor']
                self.env.scenario.seed = self.seeds['scenario']
                self.env.reset()
                self.pid.reset()
                if self.use_pid_load:
                    self.pid_load(1)
                if self.hist_init:
                    self._hist_init()
        return self.get_state(self.norm)

    def _hist_init(self):
        self.rolling = []
        env_init_dict = copy.deepcopy(self.env_init_dict)
        self.env.patient._state = env_init_dict['state']
        self.env.patient._t = env_init_dict['time']
        if self.start_date is not None:
            # need to reset date in start time
            orig_start_time = env_init_dict['time_hist'][0]
            new_start_time = datetime(year=self.start_date.year, month=self.start_date.month,
                                      day=self.start_date.day)
            new_time_hist = ((np.array(env_init_dict['time_hist']) - orig_start_time) + new_start_time).tolist()
            self.env.time_hist = new_time_hist
        else:
            self.env.time_hist = env_init_dict['time_hist']
        self.env.BG_hist = env_init_dict['bg_hist']
        self.env.CGM_hist = env_init_dict['cgm_hist']
        self.env.risk_hist = env_init_dict['risk_hist']
        self.env.LBGI_hist = env_init_dict['lbgi_hist']
        self.env.HBGI_hist = env_init_dict['hbgi_hist']
        self.env.CHO_hist = env_init_dict['cho_hist']
        self.env.insulin_hist = env_init_dict['insulin_hist']
        self.env.magni_risk_hist = env_init_dict['magni_risk_hist']

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    @property
    def action_space(self):
        return spaces.Box(low=0, high=0.1, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
        if self.gt:
            return spaces.Box(low=0, high=np.inf, shape=(len(st),))
        else:
            num_channels = int(len(st)/self.state_hist)
            return spaces.Box(low=0, high=np.inf, shape=(num_channels, self.state_hist))

