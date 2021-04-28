import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from braindecode.torch_ext.util import np_to_var


def FocalLoss(inputs, targets, alpha=None, gamma=2, size_average=True):
    # input, target, weight = None, size_average = None, ignore_index = -100,
    # reduce = None, reduction = 'mean'
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    t = Variable(torch.FloatTensor([2]))

    return t.cuda()

    N = inputs.size(0)
    C = inputs.size(1)
    valid_ch = -(nan_ch - 1)
    inputs_c = Variable(torch.FloatTensor(torch.sum(valid_ch), 2))
    targets_onehot = Variable(torch.IntTensor(torch.sum(valid_ch), 2))
    h = 0
    for i in range(N):
        cur_ch = valid_ch[i]
        for j in range(C):
            if cur_ch[j] == 1:
                inputs_c[h] = inputs[i, j]
                # targets_onehot[h, targets[i]] = 1
                targets_onehot[h] = torch.tensor([0, 1]) \
                    if targets[i] == 1 else torch.tensor([1, 0])
                h += 1

    targets_onehot = targets_onehot.cuda()
    inputs_c = inputs_c.cuda()

    batch_loss = -targets_onehot.view(-1, 1) * (1 - inputs_c.view(-1, 1)) ** gamma * torch.log(
        inputs_c.view(-1, 1) + 1e-12)

    if size_average:
        loss = batch_loss.mean()
    else:
        loss = batch_loss.sum()
    return loss


def BCEloss(input, target, size_average=False):
    input = torch.mean(input, dim=2)  # N,S,C,I => N,S,I
    input = input.permute(2, 0, 1)  # N,S,I => I,N,S
    target = target.permute(1, 0)  # N,I => I,N
    target_onehot = F.one_hot(target, num_classes=-1)  # I,N => I,N,S
    loss_list = []
    if size_average:
        for i in range(input.shape[0]):
            batch_loss = F.binary_cross_entropy(input[i], target_onehot[i].float(), reduction='mean')
            loss_list.append(batch_loss)
        loss = torch.tensor(np.array(loss_list, dtype='float32'), requires_grad=True).cuda()
    else:
        for i in range(input.shape[0]):
            batch_loss = F.binary_cross_entropy(input[i], target_onehot[i].float(), reduction='sum')
            loss_list.append(batch_loss)
        loss = torch.tensor(np.array(loss_list, dtype='float32'), requires_grad=True).cuda()
    return loss


def NLLloss(inputs, targets, nan_ch, size_average=True):
    N = inputs.size(0)
    C = inputs.size(1)
    valid_ch = -(nan_ch - 1)
    inputs_c = Variable(torch.FloatTensor(torch.sum(valid_ch), 2))
    targets_2 = Variable(torch.FloatTensor(torch.sum(valid_ch)))
    h = 0
    for i in range(N):
        cur_ch = valid_ch[i]
        for j in range(C):
            if cur_ch[j] == 1:
                inputs_c[h] = inputs[i, j]
                targets_2[h] = torch.tensor(1) \
                    if targets[i] == 1 else torch.tensor(0)
                h += 1
    targets_2 = targets_2.cuda()
    inputs_c = inputs_c.cuda()
    loss = F.nll_loss(torch.log(inputs_c), targets_2, reduction='sum')
    return loss
