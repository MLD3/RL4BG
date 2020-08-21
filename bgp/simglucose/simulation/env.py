from bgp.simglucose.patient.t1dpatient import Action
from bgp.simglucose.analysis.risk import risk_index, magni_risk_index
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
import logging
import joblib
from collections import namedtuple
from bgp.simglucose.simulation.rendering import Viewer

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple('Observation', ['CGM'])
logger = logging.getLogger(__name__)


class T1DSimEnv(object):
    # TODO: in process of removing risk_diff default and moving to platform
    def __init__(self, patient, sensor, pump, scenario, sample_time=None, model=None, model_device=None, source_dir=None):
        self.patient = patient
        self.state = self.patient.state  # caching for model usage
        # TODO: make more general
        norm_params_full = joblib.load('{}/bgp/simglucose/params/adult_001_std_params.pkl'.format(source_dir))
        new_mask = [True for _ in range(13)] + [True, False, False, True]  # throwing out BG and CGM
        norm_params_new = {'mu': norm_params_full['mu'][new_mask],
                           'std': norm_params_full['sigma'][new_mask]}
        self.norm_params = norm_params_new
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.perm_sample_time = sample_time
        self.model = model
        self.model_device = model_device
        self._reset()


    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action, cho):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        if cho is not None:
            CHO = cho
        else:
            CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun, cho):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0
        if self.model is not None:
            # Calculate CHO/insulin
            for _ in range(int(self.sample_time)):
                patient_action = self.scenario.get_action(self.time)
                tmp_basal = self.pump.basal(action.basal)
                tmp_bolus = self.pump.bolus(action.bolus)
                tmp_insulin = tmp_basal + tmp_bolus
                if cho is not None:
                    tmp_CHO = cho
                else:
                    tmp_CHO = patient_action.meal
                CHO += tmp_CHO / self.sample_time
                insulin += tmp_insulin / self.sample_time
                self.patient._t += 1  # copying mini-step of 1 minute
            # Make state
            state = np.concatenate([self.state, [CHO, insulin]])
            norm_state = ((state-self.norm_params['mu'])/self.norm_params['std']).reshape(1, -1)
            tensor_state = torch.from_numpy(norm_state).float().to(self.model_device)
            # feed through model
            with torch.no_grad():
                next_state_tensor = self.model(tensor_state)
                if self.model_device != 'cpu':
                    next_state_tensor = next_state_tensor.cpu()
                next_state_norm = next_state_tensor.numpy().reshape(-1)
            next_state = (next_state_norm*self.norm_params['std'][:13])+self.norm_params['mu'][:13]
            self.state = next_state
            # calculate BG and CGM
            BG = self.state[12]/self.patient._params.Vg
            self.patient._state[12] = self.state[12]  # getting observation correct for CGM measurement
            CGM = self.sensor.measure(self.patient)
        else:
            for _ in range(int(self.sample_time)):
                # Compute moving average as the sample measurements
                tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action, cho)
                CHO += tmp_CHO / self.sample_time
                insulin += tmp_insulin / self.sample_time
                BG += tmp_BG / self.sample_time
                CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        magni_risk = magni_risk_index([BG])
        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)
        self.magni_risk_hist.append(magni_risk)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(bg_hist=self.BG_hist, cgm_hist=self.CGM_hist, insulin_hist=self.insulin_hist,
                            risk_hist=self.risk_hist)
        done = BG < 40 or BG > 350
        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state)

    def _reset(self):
        if self.perm_sample_time is None:
            self.sample_time = self.sensor.sample_time
        else:
            self.sample_time = self.perm_sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        magni_risk = magni_risk_index([BG])
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.magni_risk_hist = [magni_risk]
        self.CHO_hist = [0]
        self.insulin_hist = [0]

    def reset(self, sensor_seed_change=True, incr_day=0):
        self.patient.reset()
        self.state = self.patient.state
        self.sensor.reset()
        if sensor_seed_change:
            self.sensor.seed = self.sensor.seed + 1
        self.pump.reset()
        self.scenario.reset()
        self.scenario.start_time = self.scenario.start_time + timedelta(days=incr_day)
        self._reset()
        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state)

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df['Magni_Risk'] = pd.Series(self.magni_risk_hist)
        df = df.set_index('Time')
        return df
