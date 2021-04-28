import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=0.5, size_average=True, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.eps = eps

    def forward(self, preds, target):
        '''
        :param input: input [N,S,I] [12,2,1]
        :param target: target [N,C] [12,2]
        :return: loss
        '''
        preds = torch.squeeze(preds, dim=2)  # N,S,I => N,S
        target = target.float()
        pt = preds[:, 0]

        loss = - target * self.alpha * (1.0 - pt) ** self.gamma * torch.log(pt + self.eps) \
               - (1-target) * (1.0 - self.alpha) * pt ** self.gamma * torch.log(1.0 - pt + self.eps)

        if self.size_average:
            return loss.mean()*5
        else:
            return loss.sum()
