import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from torch_ext.modules import Expression
from torch_ext.util import np_to_var

def identity(x):
    return x

class EDGCN(nn.Module):

    def __init__(
            self,
            in_chans,
            n_classes,
            input_time_length,
            extra_length,
            final_conv_length="auto",
            n_filters_time=15,
            filter_time_length=10,
            channel_res_conv_length=3,
            pool_time_length=3,
            pool_time_stride=3,
            n_filters_2=15,
            filter_length_2=10,
            n_filters_3=30,
            filter_length_3=10,
            n_filters_4=30,
            filter_length_4=10,
            n_filters_5=100,
            filter_length_5=10,
            first_nonlin=elu,
            first_pool_nonlin=identity,
            later_nonlin=elu,
            later_pool_nonlin=identity,
            drop_prob=0.5,
            split_first_layer=True,
            double_time_convs=False,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
    ):
        super(EDGCN, self).__init__()
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self
        self.create_network()

    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride

        conv_chans = self.n_filters_time * self.in_chans

        self.conv_time = nn.Conv1d(self.in_chans, conv_chans,
                                   kernel_size=self.filter_time_length, stride=conv_stride, groups=self.in_chans)

        self.bnorm = nn.BatchNorm1d(
            conv_chans,
            momentum=self.batch_norm_alpha,
            affine=True,
            eps=1e-5,
        )

        self.conv_nonlin = nn.LeakyReLU(0.2, inplace=True)

        self.pool = nn.MaxPool1d(
            kernel_size=self.pool_time_length, stride=pool_stride,
        )
        self.pool_nonlin = Expression(self.first_pool_nonlin)

        def add_conv_pool_block(
                n_filters_before, n_filters, filter_length, block_nr
        ):
            model = nn.Sequential()
            suffix = "_{:d}".format(block_nr)
            # model.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            model.add_module(
                "conv" + suffix,
                nn.Conv1d(
                    n_filters_before * self.in_chans,
                    n_filters * self.in_chans,
                    filter_length,
                    stride=conv_stride,
                    groups=self.in_chans,
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                model.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm1d(
                        n_filters * self.in_chans,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            # model.add_module("nonlin" + suffix, Expression(self.later_nonlin))
            model.add_module("nonlin" + suffix, nn.LeakyReLU(0.2, inplace=True))
            model.add_module(
                "pool" + suffix,
                nn.MaxPool1d(
                    kernel_size=self.pool_time_length,
                    stride=pool_stride,
                ),
            )
            model.add_module(
                "pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
            )
            param_dict = dict(list(model.named_parameters()))
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)
            return model

        self.block2 = add_conv_pool_block(
            self.n_filters_time, self.n_filters_2, self.filter_length_2, 2
        )
        self.block3 = add_conv_pool_block(
            self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        self.block4 = add_conv_pool_block(
            self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        self.drop_classifier = nn.Dropout(p=self.drop_prob)

        self.conv_channel_res = nn.Conv1d(
            in_channels=self.n_filters_4 * self.in_chans,
            out_channels=self.in_chans,
            kernel_size=self.channel_res_conv_length,
            bias=True,
        )
        self.channel_elu = Expression(elu)

        if self.final_conv_length == "auto":
            out = self.test(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_time_length),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time

        self.classifier_conv = nn.Conv1d(1, 32,
                                         kernel_size=self.in_chans * self.final_conv_length + self.extra_length)
        self.fc = nn.Linear(32, 2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        if not self.batch_norm:
            init.constant_(self.conv_time.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)

        init.xavier_uniform_(self.fc.weight, gain=1)
        init.xavier_uniform_(self.classifier_conv.weight, gain=1)
        init.constant_(self.conv_channel_res.bias, 0)
        init.constant_(self.classifier_conv.bias, 0)
        init.constant_(self.fc.bias, 0)
        return

    def test(self, x):
        x = self.conv_time(x)
        x = self.conv_nonlin(self.bnorm(x))
        x = self.pool_nonlin(self.pool(x))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.drop_classifier(x)
        x = self.conv_channel_res(x)
        x = self.channel_elu(x)
        return x

    def forward(self, x, extra):
        x = self.conv_time(x)
        x = self.conv_nonlin(self.bnorm(x))
        x = self.pool_nonlin(self.pool(x))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_channel_res(x)
        x = self.channel_elu(x)
        B, C, W = x.size()
        x = self.classifier_conv(torch.cat((x.view(B, -1), extra), dim=1).unsqueeze(dim=1))
        x = self.drop_classifier(x)
        x = self.fc(x.view(B, -1))
        x = self.log_softmax(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    model = EDGCN(5, 2, input_time_length=3000,
                  extra_length=5,
                  n_filters_time=8,
                  final_conv_length="auto", drop_prob=0.8).cpu()
    summary(model, [(5, 3000), (5,)], device='cpu')
