import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Gconv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        #
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.W = nn.Parameter(torch.Tensor(self.num_outputs, self.num_outputs))#learnable parameters
        nn.init.xavier_uniform_(self.W.data, gain=1.414)


    def forward(self, A, x, norm=True,):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        ux = self.u_fc(x)
        #
        M = torch.matmul(ax, (self.W + self.W.transpose(0, 1)) / 2)
        M = torch.matmul(M, ax.transpose(1, 2))
        M = F.normalize(M, p=1, dim=-2)
        #
        A_p = F.normalize(M + A, p=1, dim=-2)
        x = torch.bmm(A_p, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x



class Siamese_Gconv(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2

if __name__=="__main__":
    """
    """
    gconv = Gconv(3,5)
    x = torch.Tensor([[[0.1, 0.1, 0.1], [0.1, 0.2, 0.1]], [[0.2, 0.2, 0.2], [0.3, 0.1, 0.2]]])
    A = torch.randint(0,2,(2,2,2))
    A = A.type_as(x)

    y = gconv(A,x)
    print("Done")