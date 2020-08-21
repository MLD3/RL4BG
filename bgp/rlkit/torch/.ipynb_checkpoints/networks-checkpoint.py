"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""


from collections import OrderedDict
import torch
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import functional as F
import math
import torch.utils.checkpoint as cp

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class BabbyNet(PyTorchModule):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.save_init_params(locals())

    def forward(self, x):
        return self.fc(x)


class SimpleCNNQ(PyTorchModule):
    """
    Baseline 1D-CNN for Deep Q network
    TODO: this architecture isn't necessarily any good for glucose
    """
    def __init__(self, input_size, output_size, device, init_w=3e-3):
        self.save_init_params(locals())
        super(SimpleCNNQ, self).__init__()
        self.channel_size = input_size[0]
        self.signal_length = input_size[1]
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=self.channel_size, out_channels=32, kernel_size=3)),
            ('bn1_1', nn.BatchNorm1d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm1d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=2)),

            ('conv2_1', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm1d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm1d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2))
        ]))

        feature_size = self.determine_feature_size(input_size)

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=512)),
            ('bn_d', nn.BatchNorm1d(num_features=512)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2))
        ]))

        self.last_fc = nn.Linear(512, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.action_size = output_size
        self.device = device

    def determine_feature_size(self, input_size):
        with torch.no_grad():
            fake_input = Variable(torch.randn(input_size)[None, :, :])
            fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, input, action_input=None):
        if action_input is not None:
            input = input.reshape((-1, self.channel_size-1, self.signal_length))
            action_stack = tuple(action_input.flatten() for _ in range(self.signal_length))
            action_stack = torch.stack(action_stack).transpose(0, 1)[:, None, :]
            input = torch.cat((action_stack, input), dim=1)
        else:
            input = input.reshape((-1, self.channel_size, self.signal_length))
        feat = self.convolution(input)
        feat = feat.view(input.size(0), -1)
        feat = self.dense(feat)
        return self.last_fc(feat)


class FancyCNNQ(PyTorchModule):
    """
    Slightly-less Baseline 1D-CNN for Deep Q network
    TODO: this architecture isn't necessarily any good for glucose
    """
    def __init__(self, input_size, output_size, device, init_w=3e-3):
        self.save_init_params(locals())
        super(FancyCNNQ, self).__init__()
        self.channel_size = input_size[0]
        self.signal_length = input_size[1]
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=self.channel_size, out_channels=32, kernel_size=7, stride=2, padding=3)),
            ('bn1_1', nn.BatchNorm1d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
            ('conv1_2', nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm1d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2)),

            ('conv2_1', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm1d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm1d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(kernel_size=2)),

            ('conv3_1', nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)),
            ('bn3_1', nn.BatchNorm1d(num_features=128)),
            ('relu3_1', nn.ReLU()),
            ('conv3_2', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)),
            ('bn3_2', nn.BatchNorm1d(num_features=128)),
            ('relu3_2', nn.ReLU()),
            ('maxpool4', nn.MaxPool1d(kernel_size=2))
        ]))

        feature_size = self.determine_feature_size(input_size)

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=256)),
            ('bn_d', nn.BatchNorm1d(num_features=256)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2))
        ]))

        self.last_fc = nn.Linear(256, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.action_size = output_size
        self.device = device

    def determine_feature_size(self, input_size):
        with torch.no_grad():
            fake_input = Variable(torch.randn(input_size)[None, :, :])
            fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, input, action_input=None):
        if action_input is not None:
            input = input.reshape((-1, self.channel_size-1, self.signal_length))
            action_stack = tuple(action_input.flatten() for _ in range(self.signal_length))
            action_stack = torch.stack(action_stack).transpose(0, 1)[:, None, :]
            input = torch.cat((action_stack, input), dim=1)
        else:
            input = input.reshape((-1, self.channel_size, self.signal_length))
        feat = self.convolution(input)
        feat = feat.view(input.size(0), -1)
        feat = self.dense(feat)
        return self.last_fc(feat)


# TODO
# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm1d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm1d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class SimpleGRUQ(PyTorchModule):
    """
    Baseline 1D-GRU for Deep Q network
    TODO: this architecture isn't necessarily any good for glucose
    """
    def __init__(self, input_size, output_size, device, hidden_size=128, num_layers=1, init_w=3e-3):
        self.save_init_params(locals())
        super(SimpleGRUQ, self).__init__()
        self.channel_size = input_size[0]
        self.signal_length = input_size[1]
        self.features = nn.GRU(input_size=self.channel_size,
                               hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True)

        self.last_fc = nn.Linear(hidden_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.action_size = output_size
        self.device = device

    def forward(self, input, action_input=None):
        if action_input is not None:
            input = input.reshape(-1, self.channel_size-1, self.signal_length).permute(0, 2, 1)
            action_stack = tuple(action_input.flatten() for _ in range(self.signal_length))
            action_stack = torch.stack(action_stack).transpose(0, 1)[:, :, None].float()
            input = torch.cat((action_stack, input), dim=2)
        else:
            input = input.reshape(-1, self.channel_size, self.signal_length).permute(0, 2, 1)
        h, _ = self.features(input)
        feat = h[:, -1, :]
        return self.last_fc(feat)


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            device='cpu'
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.device = device
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
