import abc
from typing import Iterable

import numpy as np
import torch

from bgp.rlkit.core.rl_algorithm import RLAlgorithm
from bgp.rlkit.torch import pytorch_util as ptu
from bgp.rlkit.torch.core import PyTorchModule


class TorchRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch, self.device)

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        self.device = device
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(self.device)


def _elem_or_tuple_to_variable(elem_or_tuple, device):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return torch.from_numpy(elem_or_tuple).float().to(device)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch, device):
    return {
        k: _elem_or_tuple_to_variable(x, device)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
