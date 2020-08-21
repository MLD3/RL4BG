from .base import Patient
import numpy as np
from scipy.integrate import ode, solve_ivp
import pandas as pd
from collections import namedtuple
from typing import NamedTuple
import logging
import joblib

logger = logging.getLogger(__name__)

class Action(NamedTuple):  # This plays nicer with serialization
    CHO: float
    insulin: float


Observation = namedtuple("observation", ['Gsub'])

"""
class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5    # g/min CHO

    def __init__(self, params, init_state=None, t0=0, integrator='dopri5'):
        '''
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        '''
        self.integrator = integrator
        self._params = params
        if init_state is None:
            init_state = self._params.iloc[2:15]
        self.init_state = init_state
        self.t0 = t0
        self._t = self.t0
        self.reset()


    @classmethod
    def withID(cls, patient_id, patient_para_file, **kwargs):
        '''
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        '''
        patient_params = pd.read_csv(patient_para_file)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, patient_para_file, **kwargs):
        '''
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        '''
        patient_params = pd.read_csv(patient_para_file)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info('t = {}, patient starts eating ...'.format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True

        if to_eat > 0:
            # print(action.CHO)
            logger.debug('t = {}, patient eats {} g'.format(
                self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO   # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info('t = {}, Patient finishes eating!'.format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # ODE solver

        # print('Current simulation time: {}'.format(self.t))
        # print(self._last_Qsto)
        self._odesolver.set_f_params(
            action, self._params, self._last_Qsto, self._last_foodtaken)
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error('ODE solver failed!!')
            raise ValueError('ODE solver Failed')

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        # finding state labels
        # x_0: stomach solid
        # x_1: stomach liquid
        # x_2: gut
        # x_3: plasma glucose
        # x_4: tissue glucose
        # x_5: plasma insulin
        # x_6: insulin action on glucose utilization, X(t)
        # x_7: insulin action on glucose production, I'(t)
        # x_8: delayed insulin action on liver, X^L
        # x_9: liver insulin
        # x_10: subcutaneous insulin compartment 1, I_sc1
        # x_11: subcutaneous insulin compartment 2, I_sc2
        # x_12: subcutaneous glucose

        dxdt = np.zeros(13)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)  # equazione 27
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
            x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug('t = {}, injecting insulin: {}'.format(
                t, action.insulin))

        return dxdt

    @staticmethod
    def model_jac(t, x, action, params, last_Qsto, last_foodtaken):
        # Haven't rigorously tested
        # First calculating necessary primary information, then moving to jacobian step, could potentially feed stuff in from model output
        Dbar = last_Qsto + last_foodtaken
        qsto = x[0] + x[1]

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min

        # BEGINNING JACOBIAN
        # x_0
        dxdt_dx0 = np.zeros(13)
        dxdt_dx0[0] = -params.kmax

        # Glucose in the stomach
        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            sec1 = 1 / (np.cosh(aa * ((x[0]+x[1]) - params.b * Dbar)))
            sec2 = 1 / (np.cosh(cc * ((x[0] + x[1]) - params.d * Dbar)))
            kgut_dx0 = 0 + (params.kmax - params.kmin) / 2 * (sec1 * aa - sec2 * cc + 0)
        else:
            kgut_dx0 = 0
        dxdt_dx0[1] = params.kmax - x[1] * kgut_dx0
        dxdt_dx0[2] = kgut_dx0 * x[1]

        # x_1
        dxdt_dx1 = np.zeros(13)
        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            sec1 = 1 / (np.cosh(aa * ((x[0]+x[1]) - params.b * Dbar)))
            sec2 = 1 / (np.cosh(cc * ((x[0] + x[1]) - params.d * Dbar)))
            kgut_dx1 = 0 + (params.kmax - params.kmin) / 2 * (sec1 * aa - sec2 * cc + 0)
        else:
            kgut_dx1 = 0
        dxdt_dx1[1] = - kgut_dx1
        dxdt_dx1[2] = kgut + kgut_dx1 * x[1]
        # x_2
        dxdt_dx2 = np.zeros(13)
        dxdt_dx2[2] = - params.kabs
        if x[3] >= 0:
            dxdt_dx2[3] = params.f * params.kabs / params.BW
        # x_3
        dxdt_dx3 = np.zeros(13)
        if params.kp1 - params.kp2 * x[3] - params.kp3 * x[8] > 0:
            EGPt_dx3 = - params.kp2
        else:
            EGPt_dx3 = 0
        if x[3] > params.ke2:
            Et_dx3 = params.ke1
        else:
            Et_dx3 = 0
        if x[3] >= 0:
            dxdt_dx3[3] = EGPt_dx3 + 0 - 0 - Et_dx3 - params.k1
        if x[4] >= 0:
            dxdt_dx3[4] = params.k1
        if x[12] >= 0:
            dxdt_dx3[12] = params.ksc
        # x_4
        dxdt_dx4 = np.zeros(13)
        if x[3] >= 0:
            dxdt_dx4[3] = params.k2
        Uidt_dx4 = Vmt * Kmt / (Vmt+x[4])**2
        if x[4] >= 0:
            dxdt_dx4[4] = -Uidt_dx4 + 0 - params.k2
        # x_5
        dxdt_dx5 = np.zeros(13)
        if x[5] >= 0:
            dxdt_dx5[5] = -(params.m2 + params.m4)
        It_dx5 = 1 / params.Vi
        dxdt_dx5[6] = params.p2u * It_dx5
        dxdt_dx5[7] = params.ki * It_dx5
        if x[9] >= 0:
            dxdt_dx5[9] = params.m2
        # x_6
        dxdt_dx6 = np.zeros(13)
        if x[4] >= 0:
            dxdt_dx6[4] = x[4] * params.Vmx / (Kmt + x[4])
        dxdt_dx6[6] = -params.p2u
        # x_7
        dxdt_dx7 = np.zeros(13)
        dxdt_dx7[7] = -params.ki
        dxdt_dx7[8] = params.ki
        # x_8
        dxdt_dx8 = np.zeros(13)
        if params.kp1 - params.kp2 * x[3] - params.kp3 * x[8] > 0:
            EGPt_dx8 = - params.kp3
        else:
            EGPt_dx8 = 0
        if x[3] >= 0:
            dxdt_dx8[3] = EGPt_dx8
        dxdt_dx8[8] = -params.ki
        # x_9
        dxdt_dx9 = np.zeros(13)
        if x[5] >= 0:
            dxdt_dx9[5] = params.m1
        if x[9] >= 0:
            dxdt_dx9[9] = -(params.m1 + params.m30)
        # x_10
        dxdt_dx10 = np.zeros(13)
        if x[5] >= 0:
            dxdt_dx10[5] = params.ka1
        if x[10] >= 0:
            dxdt_dx10[10] = - (params.ka1 + params.kd)
        if x[11] >= 0:
            dxdt_dx10[11] = params.kd
        # x_11
        dxdt_dx11 = np.zeros(13)
        if x[5] >= 0:
            dxdt_dx11[5] = params.ka2
        if x[11] >= 0:
            dxdt_dx11[11] = - params.ka2
        # x_12
        dxdt_dx12 = np.zeros(13)
        if x[12] >= 0:
            dxdt_dx12[12] = -params.ksc
        return (dxdt_dx0, dxdt_dx1, dxdt_dx2, dxdt_dx3, 
                dxdt_dx4, dxdt_dx5, dxdt_dx6, dxdt_dx7, 
                dxdt_dx8, dxdt_dx9, dxdt_dx10, dxdt_dx11, dxdt_dx12)


    @property
    def observation(self):
        '''
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        '''
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator(self.integrator)
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0
"""

class T1DPatientNew(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(self, params, init_state=None, t0=0, integrator='LSODA'):
        '''
        T1DPatient constructor.
        Inputs:
            - params: a pandas sequence
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.iloc[2:15]
            - t0: simulation start time, it is 0 by default
        '''
        self.integrator = integrator
        self._params = params
        if init_state is None:
            init_state = self._params.iloc[2:15]
        self.init_state = np.array(init_state, dtype=float)
        self.t0 = t0
        self._t = self.t0
        self.reset()

    @classmethod
    def withID(cls, patient_id, patient_para_file, **kwargs):
        '''
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        '''
        patient_params = pd.read_csv(patient_para_file)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, patient_para_file, **kwargs):
        '''
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        '''
        patient_params = pd.read_csv(patient_para_file)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._state

    @property
    def t(self):
        return self._t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info('t = {}, patient starts eating ...'.format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True

        if to_eat > 0:
            # print(action.CHO)
            logger.debug('t = {}, patient eats {} g'.format(
                self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info('t = {}, Patient finishes eating!'.format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # ODE solver

        # print('Current simulation time: {}'.format(self.t))
        # print(self._last_Qsto)
        sol = solve_ivp(fun=lambda time, state: self.model(time, state, action, self._params,
                                                           self._last_Qsto, self._last_foodtaken),
                        t_span=(self.t, self.t+self.sample_time),
                        y0=self.state,
                        method=self.integrator,
                        rtol=10)
        self._state = np.array(sol.y[:, -1], dtype=float)
        self._t = sol.t[-1]
        self.sol = sol
        if not sol.success:
            raise ValueError('Integrator Failed')

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        # finding state labels
        # x_0: stomach solid
        # x_1: stomach liquid
        # x_2: gut
        # x_3: plasma glucose
        # x_4: tissue glucose
        # x_5: plasma insulin
        # x_6: insulin action on glucose utilization, X(t)
        # x_7: insulin action on glucose production, I'(t)
        # x_8: delayed insulin action on liver, X^L
        # x_9: liver insulin
        # x_10: subcutaneous insulin compartment 1, I_sc1
        # x_11: subcutaneous insulin compartment 2, I_sc2
        # x_12: subcutaneous glucose

        dxdt = np.zeros(13)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)  # equazione 27
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
                  x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        if action.insulin > basal:
            logger.debug('t = {}, injecting insulin: {}'.format(
                t, action.insulin))

        return dxdt

    @property
    def observation(self):
        '''
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        '''
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._t = self.t0
        self._state = self.init_state

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter(
        '%(name)s: %(levelname)s: %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    p = T1DPatientNew.withName('adolescent#001')
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()
