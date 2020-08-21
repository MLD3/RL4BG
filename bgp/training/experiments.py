"""
Programatic interface for experiments
"""
import joblib
import os
import time
import multiprocessing as mp
import numpy as np
from bgp.rl.helpers import ModelConfig

def flatten_list(l):
    return np.concatenate([item for sublist in l for item in sublist])


class RunManager:
    def __init__(self, device_list, save_path, run_func):
        self.device_names = device_list
        self.device_free = [True for _ in device_list]
        self.device_process = [None for _ in device_list]
        self.process_stack = []
        self.save_loc = save_path
        self.run_func = run_func

    def add_job(self, run_config):
        load_path = os.path.join(self.save_loc, run_config['name'])+'.pkl'
        joblib.dump(run_config, load_path)
        self.process_stack.append(load_path)

    def done(self):
        free_device = sum(self.device_free)
        num_jobs = len(self.process_stack)
        print('------------------------------------')
        print('{}/{} devices free, {} tasks remain'.format(free_device, len(self.device_free), num_jobs))
        print('------------------------------------')
        return np.all(self.device_free) and len(self.process_stack) == 0

    def update_free(self):
        for i in range(len(self.device_process)):
            if self.device_process[i] is not None:
                if not self.device_process[i].is_alive():
                    print('job on device {} not alive'.format(self.device_names[i]))
                    self.device_process[i] = None
                    self.device_free[i] = True
            else:
                pass

    def launch_if_able(self):
        for i in range(len(self.device_names)):
            if self.device_free[i]:
                if len(self.process_stack) == 0:
                    print('No more jobs to launch')
                    break
                load_path = self.process_stack.pop()
                device = self.device_names[i]
                self.device_process[i] = mp.Process(target=self.run_func, args=(load_path, device))
                self.device_free[i] = False
                print('Launching job at {} on process {}'.format(load_path, device))
                self.device_process[i].start()

    def run_until_empty(self, sleep_interval=10):
        while not self.done():
            self.launch_if_able()
            time.sleep(sleep_interval)
            self.update_free()


class RLRunManager:
    def __init__(self, device_list, run_func, run_type, full_save):
        self.device_names = device_list
        self.device_free = [True for _ in device_list]
        self.device_process = [None for _ in device_list]
        self.process_stack = []
        self.run_func = run_func
        self.run_type = run_type
        self.full_save = full_save

    def add_job(self, run_config):
        self.process_stack.append(run_config)

    def done(self):
        free_device = sum(self.device_free)
        num_jobs = len(self.process_stack)
        print('------------------------------------')
        print('{}/{} devices free, {} tasks remain'.format(free_device, len(self.device_free), num_jobs))
        print('------------------------------------')
        return np.all(self.device_free) and len(self.process_stack) == 0

    def update_free(self):
        for i in range(len(self.device_process)):
            if self.device_process[i] is not None:
                if not self.device_process[i].is_alive():
                    print('job on device {} not alive'.format(self.device_names[i]))
                    self.device_process[i] = None
                    self.device_free[i] = True
            else:
                pass

    def launch_if_able(self):
        for i in range(len(self.device_names)):
            if self.device_free[i]:
                if len(self.process_stack) == 0:
                    print('No more jobs to launch')
                    break
                run_config = self.process_stack.pop()
                device = self.device_names[i]
                # This is kind of stupid
                mcdict = run_config.model_config._asdict()
                mcdict['device'] = device
                new_model_config = ModelConfig(**mcdict)
                self.device_process[i] = mp.Process(target=self.run_func, args=(run_config.env_config,
                                                                                new_model_config,
                                                                                run_config.train_config,
                                                                                self.run_type,
                                                                                self.full_save))
                self.device_free[i] = False
                print('Launching job on process {}'.format(device))
                self.device_process[i].start()

    def run_until_empty(self, sleep_interval=10):
        while not self.done():
            self.launch_if_able()
            time.sleep(sleep_interval)
            self.update_free()


class RLKitRunManager:
    def __init__(self, device_list, run_func):
        self.device_names = device_list
        self.device_free = [True for _ in device_list]
        self.device_process = [None for _ in device_list]
        self.process_stack = []
        self.run_func = run_func

    def add_job(self, run_tup):
        self.process_stack.append(run_tup)

    def done(self):
        free_device = sum(self.device_free)
        num_jobs = len(self.process_stack)
        print('------------------------------------')
        print('{}/{} devices free, {} tasks remain'.format(free_device, len(self.device_free), num_jobs))
        print('------------------------------------')
        return np.all(self.device_free) and len(self.process_stack) == 0

    def update_free(self):
        for i in range(len(self.device_process)):
            if self.device_process[i] is not None:
                if not self.device_process[i].is_alive():
                    print('job on device {} not alive'.format(self.device_names[i]))
                    self.device_process[i] = None
                    self.device_free[i] = True
            else:
                pass

    def launch_if_able(self):
        for i in range(len(self.device_names)):
            if self.device_free[i]:
                if len(self.process_stack) == 0:
                    print('No more jobs to launch')
                    break
                run_config, run_func = self.process_stack.pop()
                device = self.device_names[i]
                # This is stupid
                if 'algo_params' in run_config:
                    run_config['algo_params']['device'] = device
                run_config['device'] = device
                try:
                    self.device_process[i] = mp.Process(target=run_func, args=(run_config,))
                except:
                    print('wooo')
                self.device_free[i] = False
                print('Launching job on process {}'.format(device))
                self.device_process[i].start()

    def run_until_empty(self, sleep_interval=10):
        while not self.done():
            self.launch_if_able()
            time.sleep(sleep_interval)
            self.update_free()

if __name__ == '__main__':
    print('todo')
