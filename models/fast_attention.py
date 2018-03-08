import torch
import torch.nn as nn
from torch.autograd import Variable

from .seq2seq import Seq2Seq


class FastAttention(Seq2Seq):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        """Initialize model."""
        super(FastAttention, self).__init__(opt, src_vocab_size, trg_vocab_size)
        assert self.trg_hidden_dim == self.src_hidden_dim

        self.encoder = nn.LSTM(
            self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            self.nlayers_trg,
            batch_first=True,
            dropout=self.dropout
        )

        # overwrite decoder2vocab
        self.decoder2vocab = nn.Linear(2 * self.trg_hidden_dim, self.trg_vocab_size)

        self.init_weights()

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_init_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )  # bsize x seqlen x dim

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.view(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.view(
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)
                )
            )
        )  # bsize x seqlen x dim

        # Fast Attention dot product

        # bsize x seqlen_src x seq_len_trg
        alpha = torch.bmm(src_h, trg_h.transpose(1, 2))
        # bsize x seq_len_trg x dim
        alpha = torch.bmm(alpha.transpose(1, 2), src_h)
        # bsize x seq_len_trg x (2 * dim)
        trg_h_reshape = torch.cat((trg_h, alpha), 2)

        trg_h_reshape = trg_h_reshape.view(
            trg_h_reshape.size(0) * trg_h_reshape.size(1),
            trg_h_reshape.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit


