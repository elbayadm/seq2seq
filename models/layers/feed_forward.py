import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class feed_forward(nn.Module):
    """
    A simple linear layer with an activation
    """
    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 bias_zero=True,
                 activ=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias_zero = bias_zero
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)

        if activ is None:
            self.activ = lambda x: x
        else:
            self.activ = getattr(F, activ)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            if self.bias_zero:
                self.bias.data.zero_()
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.activ(F.linear(input, self.weight, self.bias))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) + ')'
