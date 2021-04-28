import numpy as np
import torch
from torch import nn
from torch.nn import init


# def initialize_weights(net_l):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv1d):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias.data, 0.0)


def initialize_weights(net_l):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_params(model, batch_norm=True):
    param_dict = dict(list(model.named_parameters()))
    conv_weight = param_dict["conv.weight"]
    init.xavier_uniform_(conv_weight, gain=1)
    if not batch_norm:
        conv_bias = param_dict["conv.bias"]
        init.constant_(conv_bias, 0)
    else:
        bnorm_weight = param_dict["bnorm.weight"]
        bnorm_bias = param_dict["bnorm.bias"]
        init.constant_(bnorm_weight, 1)
        init.constant_(bnorm_bias, 0)


def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).
    
    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)




