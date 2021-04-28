import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu
from .utils import identity, Expression


class DGCN(nn.Module):

    def __init__(
            self,
            opt=None
    ):
        super(DGCN, self).__init__()
        self.opt = opt

        if self.opt.stride_before_pool:
            conv_stride = self.opt.pool_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.opt.pool_stride

        conv_chans = self.opt.n_filters * self.opt.input_nc

        self.conv_time = nn.Conv1d(self.opt.input_nc, conv_chans,
                                   kernel_size=self.opt.filter_length, stride=conv_stride,
                                   groups=self.opt.input_nc)
        self.bnorm = nn.BatchNorm1d(conv_chans, momentum=0.1, affine=True, eps=1e-5, )
        self.conv_nonlin = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.block2 = self.add_conv_pool_block(
            self.opt.n_filters, self.opt.n_filters_2, self.opt.filter_length_2, 2
        )
        self.block3 = self.add_conv_pool_block(
            self.opt.n_filters_2, self.opt.n_filters_3, self.opt.filter_length_3, 3
        )
        self.block4 = self.add_conv_pool_block(
            self.opt.n_filters_3, self.opt.n_filters_4, self.opt.filter_length_4, 4
        )

        self.conv_channel_res = nn.Conv1d(
            in_channels=self.opt.n_filters_4 * self.opt.input_nc,
            out_channels=self.opt.input_nc,
            kernel_size=self.opt.channel_res_conv_length,
            bias=True,
        )
        self.channel_elu = Expression(elu)
        out = self.test(
            torch.ones(
                (1, self.opt.input_nc, self.opt.input_length), dtype=torch.float32, requires_grad=False)
        )
        n_out_time = out.cpu().data.numpy().shape[2]
        self.final_conv_length = n_out_time
        self.dropout = nn.Dropout(self.opt.drop_prob)
        self.conv_classifier = nn.Conv1d(
            in_channels=self.opt.input_nc,
            out_channels=self.opt.n_classes,
            kernel_size=self.final_conv_length,
            bias=True,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def add_conv_pool_block(self, n_filters_before, n_filters, filter_length, block_nr):
        model = nn.Sequential()
        suffix = "_{:d}".format(block_nr)
        model.add_module(
            "conv" + suffix,
            nn.Conv1d(
                n_filters_before * self.opt.input_nc,
                n_filters * self.opt.input_nc,
                filter_length,
                stride=1,
                groups=self.opt.input_nc,
                bias=not self.opt.batch_norm,
            ),
        )
        if self.opt.batch_norm:
            model.add_module("bnorm" + suffix,
                             nn.BatchNorm1d(n_filters * self.opt.input_nc, momentum=0.1, affine=True, eps=1e-5))

        model.add_module("nonlin" + suffix, nn.LeakyReLU(0.2, inplace=True))
        model.add_module("pool" + suffix, nn.MaxPool1d(kernel_size=3, stride=3))
        return model

    def test(self, x):
        x = self.conv_nonlin(self.bnorm(self.conv_time(x)))
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_channel_res(x)
        x = self.channel_elu(x)
        return x

    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv_nonlin(self.bnorm(x))
        x = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_channel_res(x)
        x = self.channel_elu(x)
        x = self.dropout(x)
        x = self.conv_classifier(x)
        x = self.log_softmax(x)
        return x

