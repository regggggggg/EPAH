import torch.nn as nn
import torch
from torch.autograd import Variable

class EPAHLoss(nn.Module):
    def __init__(self, code_length, num_train, mu):
        super(EPAHLoss, self).__init__()
        self.code_length = code_length
        self.num_train = num_train
        self.mu = mu

    def forward(self, u, g, V, S, S_1, V_omega, W, L):
        batch_size = u.size(0)
        V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
        V_omega = Variable(torch.from_numpy(V_omega).type(torch.FloatTensor).cuda())
        g = Variable(torch.from_numpy(g).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())
        S_1 = Variable(S_1.cuda())
        square_loss = (u.mm(V.t())-self.code_length * S) ** 2
        square_loss_1 = self.mu * (u.mm(g.t()) - self.code_length * S_1) **2
        label_loss = (u.mm(W) - L) ** 2
        loss = (square_loss.sum() + label_loss.sum() + square_loss_1.sum()) / (self.num_train * batch_size)
        return loss
