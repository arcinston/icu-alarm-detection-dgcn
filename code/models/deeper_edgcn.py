import numpy as np
import torch
from torch import nn
from torch_ext.util import np_to_var
from models.util import initialize_weights
from options.default_opt import Opt


class DeeperEDGCN(nn.Module):

    def __init__(
            self,
            opt=None
    ):
        super(DeeperEDGCN, self).__init__()
        self.opt = opt

        conv_chans = 8 * self.opt.input_nc

        self.conv_time1 = self.base_layer(opt.input_nc, conv_chans, opt.input_nc)
        self.conv_time2 = self.base_layer(conv_chans, 2 * conv_chans, opt.input_nc)
        self.conv_time3 = self.base_layer(2 * conv_chans, 2 * conv_chans, opt.input_nc)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(p=0.3)

        self.block21 = self.base_layer(2 * conv_chans, 2 * conv_chans, opt.input_nc)
        self.block22 = self.base_layer(2 * conv_chans, 4 * conv_chans, opt.input_nc)
        self.block23 = self.base_layer(4 * conv_chans, 4 * conv_chans, opt.input_nc)

        self.block31 = self.base_layer(4 * conv_chans, 4 * conv_chans, opt.input_nc)
        self.block32 = self.base_layer(4 * conv_chans, 8 * conv_chans, opt.input_nc)
        self.block33 = self.base_layer(8 * conv_chans, 8 * conv_chans, opt.input_nc)

        self.block41 = self.base_layer(8 * conv_chans, 8 * conv_chans, opt.input_nc)
        self.block42 = self.base_layer(8 * conv_chans, 16 * conv_chans, opt.input_nc)
        self.block43 = self.base_layer(16 * conv_chans, 8 * conv_chans, opt.input_nc)

        self.conv_channel_res = self.base_layer(8 * conv_chans, opt.input_nc, groups=1)

        out = self.test(
            np_to_var(
                np.ones(
                    (1, self.opt.input_nc, self.opt.input_length),
                    dtype=np.float32,
                )
            )
        )
        n_out_time = out.cpu().data.numpy().shape[2]
        self.final_conv_length = n_out_time

        self.conv_classifier = nn.Conv1d(
            in_channels=opt.input_nc,
            out_channels=self.opt.n_classes,
            kernel_size=self.final_conv_length,
            bias=True,
        )
        self.conv_classifier = nn.Conv1d(1, 32,
                                         kernel_size=opt.input_nc * self.final_conv_length + opt.extra_length)
        self.fc = nn.Linear(32, 2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        initialize_weights(self)
        return

    def base_layer(self, inchans, outchans, groups):
        model = []
        model += [nn.Conv1d(inchans, outchans, kernel_size=3, stride=1, groups=groups),
                  nn.BatchNorm1d(outchans),
                  nn.LeakyReLU(0.2, inplace=True)]
        net = nn.Sequential(*model)
        initialize_weights(net)
        return net

    def test(self, x):
        x = self.conv_time3(self.conv_time2(self.conv_time1(x)))
        x = self.dropout(self.pool(x))
        x = self.block23(self.block22(self.block21(x)))
        x = self.dropout(self.pool(x))
        x = self.block33(self.block32(self.block31(x)))
        x = self.dropout(self.pool(x))
        x = self.block43(self.block42(self.block41(x)))
        x = self.conv_channel_res(x)
        return x

    def forward(self, x, extra):
        x = self.conv_time3(self.conv_time2(self.conv_time1(x)))
        x = self.dropout(self.pool(x))
        x = self.block23(self.block22(self.block21(x)))
        x = self.dropout(self.pool(x))
        x = self.block33(self.block32(self.block31(x)))
        x = self.dropout(self.pool(x))
        x = self.block43(self.block42(self.block41(x)))
        x = self.conv_channel_res(x)
        B, C, W = x.size()
        x = self.conv_classifier(torch.cat((x.view(B, -1), extra), dim=1).unsqueeze(dim=1))
        x = self.fc(x.view(B, -1))
        x = self.log_softmax(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = False
    opt.use_extra = True
    opt.add_noise_prob = 0
    opt.window_size = 3000
    opt.extra_length = 5
    opt.input_nc = 5
    opt.input_length = 3000
    model = DeeperEDGCN(opt).cpu()
    summary(model, [(5, 3000), (5,)], device='cpu')
