"""
Torch argmax policy
"""
import numpy as np
import bgp.rlkit.torch.pytorch_util as ptu
from bgp.rlkit.policies.base import SerializablePolicy
from bgp.rlkit.torch.core import PyTorchModule
import torch


class ArgmaxDiscretePolicy(PyTorchModule, SerializablePolicy):
    def __init__(self, qf, device='cpu'):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float().to(self.qf.device)
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}
