import numpy as np
from tqdm import tqdm


class PID:
    def __init__(self, setpoint, kp, ki, kd, basal=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.basal = basal
        self.setpoint = setpoint

    def step(self, value):
        error = self.setpoint - value
        p_act = self.kp * error
        # print('p: {}'.format(p_act))
        self.integral += error
        i_act = self.ki * self.integral
        # print('i: {}'.format(i_act))
        d_act = self.kd * (error - self.previous_error)
        try:
            if self.basal is not None:
                b_act = self.basal
            else:
                b_act = 0
        except:
            b_act = 0
        # print('d: {}'.format(d_act))
        self.previous_error = error
        action = p_act + i_act + d_act + b_act
        return action

    def reset(self):
        self.integral = 0
        self.previous_error = 0


def pid_test(pid, env, n_days, seed, full_save=False):
    env.seeds['sensor'] = seed
    env.seeds['scenario'] = seed
    env.reset()
    full_patient_state = []
    for i in tqdm(range(n_days*288)):
        act = pid.step(env.env.CGM_hist[-1])
        state, reward, done, info = env.step(action=act)
        full_patient_state.append(info['patient_state'])
    full_patient_state = np.stack(full_patient_state)
    if full_save:
        return env.env.show_history(), full_patient_state
    else:
        return {'hist': env.env.show_history()[288:], 'kp': pid.kp, 'ki': pid.ki, 'kd': pid.kd}
