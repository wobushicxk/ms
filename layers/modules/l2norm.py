#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn


class L2Norm(nn.Cell):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = Parameter(Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        #init.constant(self.weight,self.gamma)
        self.weight.fill(self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        #x = torch.div(x,norm)
        op = ops.Div()
        x = op(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


        