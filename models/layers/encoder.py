import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class text_encoder(nn.Module):
    """A recurrent encoder with embedding layer."""
    def __init__(self, opt, src_vocab_size):
        super().__init__()
        self.rnn_type = opt.rnn_type_src.upper()
        self.input_size = opt.dim_word_src
        self.hidden_size = opt.rnn_size_src
        self.ctx_size = self.hidden_size * 2
        self.num_layers = opt.num_layers_src
        self.bidirectional = opt.bidirectional
        self.vocab_size = src_vocab_size

        # For dropout btw layers, only effective if num_layers > 1
        self.dropout_rnn = opt.encoder_dropout

        # Our other custom dropouts after embeddings and annotations
        self.dropout_emb = opt.input_encoder_dropout
        self.dropout_ctx = opt.input_encoder_dropout

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(self.dropout_emb)
        if self.dropout_ctx > 0:
            self.do_ctx = nn.Dropout(self.dropout_ctx)

        # Create embedding layer
        self.emb = nn.Embedding(self.vocab_size,
                                self.input_size,
                                padding_idx=0)  # does it serve anything

        # Create encoder according to the requested type
        RNN = getattr(nn, self.rnn_type)
        self.core = RNN(self.input_size,
                        self.hidden_size,
                        self.num_layers,
                        bias=True,
                        batch_first=True,
                        dropout=self.dropout_rnn,
                        bidirectional=self.bidirectional)

    def forward(self, x, lengths):
        """
        Receives a Variable of indices (N, T)
        and returns:
            their recurrent representations (N, T, H)
            the final state (N, H) (not needed)
            and a mask to discard the padding (N, T)
        """
        embs = self.emb(x)
        if self.dropout_emb > 0:
            embs = self.do_emb(embs)

        # Pack and encode
        packed_emb = pack_padded_sequence(embs,
                                          lengths,
                                          batch_first=True)

        # last_state is a tuple of (h_t, c_t) for LSTM or h_t for GRU
        packed_hs, _ = self.core(packed_emb)
        ctx = pad_packed_sequence(packed_hs, batch_first=True)[0]
        if self.dropout_ctx > 0:
            ctx = self.do_ctx(ctx)
        return ctx, (x != 0).float()
