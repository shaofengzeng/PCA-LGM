#-*-coding:utf-8-*-
import torch
from torch import nn
import torch.nn.functional as F
import math
import copy


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    ##method1
    d_k = query.size(-1)
    s = torch.einsum('bhnd,bhmd->bhnm', query, key) / d_k ** .5
    prob = torch.nn.functional.softmax(s, dim=-1)
    if dropout !=None:
        prob = dropout(prob)
    vec = torch.einsum('bhnm,bhmd->bhnd', prob, value)
    return vec, prob

    ##method2
    # scores = torch.matmul(query, key.transpose(-2, -1)) \
    #          / math.sqrt(d_k)
    #
    # p_attn = F.softmax(scores, dim=-1)
    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    # v_attn = torch.matmul(p_attn, value)
    # return v_attn, p_attn
    #

def clones(module, N):
    "generate n layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    """"""
    def __init__(self, head_num, d_model, dropout=0.1):
        "初始化时指定头数h和模型维度d_model"
        super(MultiHeadedAttention, self).__init__()
        #
        assert (d_model % head_num == 0)
        # 按照文中的简化，我们让d_v与d_k相等
        self.d_k = d_model // head_num
        self.h = head_num
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        "multi-head attention model"
        nbatches = query.size(0)
        #
        residual = query

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, attn = attention(query,
                                 key,
                                 value,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        #
        x += residual

        x = self.layer_norm(x)
        return x, attn

if __name__=="__main__":
    q = torch.rand((1,6,8))
    k = q
    v = q
    mha = MultiHeadedAttention(2,8)
    q,p = mha(q,k,v)
    print(q.shape)
    print(p)
    print("Done")




