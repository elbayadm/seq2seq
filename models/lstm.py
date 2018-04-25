import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np


class ConvAttention(nn.Module):
    """
    Convolutional attention
    @inproceedings{allamanis2016convolutional,
          title={A Convolutional Attention Network for Extreme Summarization of Source Code},
          author={Allamanis, Miltiadis and Peng, Hao and Sutton, Charles},
          booktitle={International Conference on Machine Learning (ICML)},
          year={2016}
      }
    """

    def __init__(self, dim, opt):
        """Initialize layer."""
        super(ConvAttention, self).__init__()
        src_emb_dim = opt.dim_word_src
        interm_dim = src_emb_dim // 2  # k1
        dim = opt.rnn_size_trg  # k2
        w1 = 25  # first kernel width
        w2 = 29  # second kernel width
        w3 = 11
        # check if padding necessary
        # padding to maintaing the same length
        self.conv1 = nn.Conv1d(src_emb_dim, interm_dim, w1, padding=(w1-1)/2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(interm_dim, dim, w2, padding=(w2-1)/2)
        self.conv3 = nn.Conv1d(dim, 1, w3, padding=(w3-1)/2)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim + src_emb_dim, dim, bias=False)

    def forward(self, input, context, src_emb):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        print('src_emb:', src_emb.size())
        L1 = self.relu(self.conv1(src_emb))
        print('L1:', L1.size())
        L2 = self.conv2(L1)
        print('L2:', L2.size())
        # columnwize dot product
        print('input:', input.size())
        L2 = L2 * input
        # L2 normalization:
        norm = L2.norm(p=2, dim=2, keepdim=True)
        print('L2 norm:', norm)
        L2 = L2.div(norm)
        attn = self.conv3(L2)
        print('attn:', attn.size())
        attn_sm = self.sm(attn)
        print('attn_sm:', attn_sm)
        attn_reshape = attn_sm.view(attn_sm.size(0), 1, attn_sm.size(1))  # batch x 1 x sourceL
        print('attn_reshape:', attn_reshape)
        weighted_context = torch.bmm(attn_reshape, src_emb).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Effective approaches to attention based NMT (Luong et al. EMNLP 15)
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, opt):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        dim = opt.rnn_size_trg
        dropout = opt.attention_dropout

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.mask = None

    def forward(self, input, context, src_emb=None):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(self.dropout(attn))
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttention(nn.Module):
    """
    A long short-term memory (LSTM) cell with attention.
    Use SoftDotAttention
    """

    def __init__(self, opt):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        # Params:
        self.mode = opt.attention_mode

        self.input_size = opt.dim_word_trg
        self.hidden_size = opt.rnn_size_trg

        self.input_weights = nn.Linear(self.input_size,
                                       4 * self.hidden_size)
        self.hidden_weights = nn.Linear(self.hidden_size,
                                        4 * self.hidden_size)

        if self.mode == "dot":
            self.attention_layer = SoftDotAttention(opt)
        elif self.mode == "conv":
            self.attention_layer = ConvAttention(opt)
        else:
            raise ValueError('Unkown attention mode %s' % self.mode)


    def forward(self, input, hidden, ctx, src_emb):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            # print('In front of the gates:')
            # print('input:', input.size())
            # print('hx:', hx.size())
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx, src_emb)

            return h_tilde, cy

        input = input.transpose(0, 1)
        output = []
        steps = list(range(input.size(0)))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                h = hidden[0]
            else:
                h = hidden
            output.append(h)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden

