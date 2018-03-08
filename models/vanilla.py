import torch
import torch.nn as nn
from .seq2seq import Seq2Seq


class Vanilla(Seq2Seq):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        """Initialize model."""
        super(Vanilla, self).__init__(opt, src_vocab_size, trg_vocab_size)
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
            dropout=self.dropout,
            batch_first=True
        )

        self.init_weights()

    def forward(self, input_src, input_trg, ctx_mask=None, trg_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_init_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

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
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )

        return decoder_logit


