from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)
CONTROL_QUEST = '/source/dir/simglucose/params/Quest.csv'
PATIENT_PARA_FILE = '/source/dir/simglucose/params/vpatient_params.csv'
ParamTup = namedtuple('ParamTup', ['basal', 'cf', 'cr'])

class BBController(Controller):
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')

        meal = kwargs.get('meal')

        action = self._bb_policy(
            pname,
            meal,
            observation.CGM,
            sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        if any(self.quest.Name.str.match(name)):
            q = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = np.asscalar(params.u2ss.values)
            BW = np.asscalar(params.BW.values)
        else:
            q = pd.DataFrame([['Average', 13.5, 23.52, 50, 30]],
                             columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43
            BW = 57.0

        basal = u2ss * BW / 6000
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.debug('glucose = {}'.format(glucose))
            bolus = np.asscalar(meal / q.CR.values + (glucose > 150)
                                * (glucose - self.target) / q.CF.values)
        else:
            bolus = 0

        bolus = bolus / env_sample_time
        action = Action(basal=basal, bolus=bolus)
        return action

    def reset(self):
        pass


class ManualBBController(Controller):
    def __init__(self, target, cr, cf, basal, sample_rate=5, use_cf=True, use_bol=True, cooldown=0,
                 corrected=True, use_low_lim=False, low_lim=70):
        super().__init__(self)
        self.target = target
        self.orig_cr = self.cr = cr
        self.orig_cf = self.cf = cf
        self.orig_basal = self.basal = basal
        self.sample_rate = sample_rate
        self.use_cf = use_cf
        self.use_bol = use_bol
        self.cooldown = cooldown
        self.last_cf = np.inf
        self.corrected = corrected
        self.use_low_lim = low_lim
        self.low_lim = low_lim

    def increment(self, cr_incr=0, cf_incr=0, basal_incr=0):
        self.cr += cr_incr
        self.cf += cf_incr
        self.basal += basal_incr

    def policy(self, observation, reward, done, **kwargs):
        carbs = kwargs.get('carbs')
        glucose = kwargs.get('glucose')
        action = self.manual_bb_policy(carbs, glucose)
        return action

    def manual_bb_policy(self, carbs, glucose, log=False):
        if carbs > 0:
            if self.corrected:
                carb_correct = carbs / self.cr
            else:
                # assuming carbs are already multiplied by sampling rate
                carb_correct = (carbs/self.sample_rate) / self.cr  # TODO: not sure about this
            hyper_correct = (glucose > self.target) * (glucose - self.target) / self.cf
            hypo_correct = (glucose < self.low_lim) * (self.low_lim - glucose) / self.cf
            bolus = 0
            if self.use_low_lim:
                bolus -= hypo_correct
            if self.use_cf:
                if self.last_cf > self.cooldown and hyper_correct > 0:
                    bolus += hyper_correct
                    self.last_cf = 0
            if self.use_bol:
                bolus += carb_correct
            bolus = bolus / self.sample_rate
        else:
            bolus = 0
            carb_correct = 0
            hyper_correct = 0
            hypo_correct = 0
        self.last_cf += self.sample_rate
        if log:
            return Action(basal=self.basal, bolus=bolus), hyper_correct, hypo_correct, carb_correct
        else:
            return Action(basal=self.basal, bolus=bolus)

    def get_params(self):
        return ParamTup(basal=self.basal, cf=self.cf, cr=self.cr)

    def adjust(self, basal_adj, cr_adj):
        self.basal += self.orig_basal * basal_adj
        self.cr += self.orig_cr * cr_adj

    def reset(self):
        self.cr = self.orig_cr
        self.cf = self.orig_cf
        self.basal = self.orig_basal
        self.last_cf = np.inf

class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        self.state = observation
        action = Action(basal=0, bolus=0)
        return action

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state
