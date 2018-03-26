import torch
import torch.nn as nn
import torch.nn.functional as F
from .feed_forward import feed_forward
from .utils import to_var
from .attention_lstm import Attention


class cond_decoder(nn.Module):
    """
    A conditional decoder with attention à la dl4mt-tutorial.
    """
    def __init__(self, opt, trg_vocab_size):
        super().__init__()
        self.input_size = opt.dim_word_trg
        self.hidden_size = opt.rnn_size_trg
        self.ctx_size = 2 * opt.rnn_size_src
        self.vocab_size = trg_vocab_size
        self.tied_decoder = opt.tied_decoder
        self.dec_init = 'mean_ctx'  # zero
        self.att_type = 'mlp'
        self.dropout_out = opt.decoder_dropout

        # Create target embeddings
        self.emb = nn.Embedding(self.vocab_size,
                                self.input_size,
                                padding_idx=0)

        # Create attention layer
        self.att = Attention(self.ctx_size,
                             self.hidden_size,
                             att_type=self.att_type)

        # Decoder initializer
        if self.dec_init == 'mean_ctx':
            self.map_dec_init = feed_forward(self.ctx_size,
                                             self.hidden_size,
                                             activ='tanh')
        # Create first decoder layer necessary for attention
        self.dec0 = nn.GRUCell(self.input_size, self.hidden_size)
        self.dec1 = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = feed_forward(self.hidden_size,
                                    self.input_size,
                                    bias_zero=True,
                                    activ='tanh')

        # Final softmax
        self.out2prob = feed_forward(self.input_size,
                                     self.vocab_size)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_decoder:
            self.out2prob.weight = self.emb.weight

    def f_init(self, ctx, ctx_mask):
        """
        Given ctx: the various hidden states of the source sequence (N, T, 2xH)
        and the source_code i.e. the final hidden state
        Returns the initial h_0 for the decoder.
        """
        if self.dec_init == 'zero':
            h_0 = torch.zeros(ctx.size(0), self.hidden_size)
            return to_var(h_0, requires_grad=False)
        elif self.dec_init == 'mean_ctx':
            return self.map_dec_init(ctx.sum(1) / ctx_mask.sum(1).unsqueeze(1))


    def f_next(self, ctx, y, h):
        h1 = self.dec0(y, h)
        # Apply attention
        alpha_t, z_t = self.att(h1.unsqueeze(0), ctx)
        h2 = self.dec1(z_t, h1)
        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)
        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)
        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit)) ## negative in the original code

        # Return log probs and new hidden states
        return log_p, h2

    def forward(self, ctx, ctx_mask, target):
        """
        Computes the softmax outputs given source annotations `ctx` (N, T, 2xH)
        and the ground-truth target token indices `target` (N, T)
        """
        #
        seq_length = target.size(1)
        y_emb = self.emb(target)
        # Get initial hidden state
        h = self.f_init(ctx, ctx_mask)
        logits = []
        # -1: So that we skip the timestep where input is <eos>
        for t in range(seq_length):
            log_p, h = self.f_next(ctx, y_emb[:,t], h)
            logits.append(log_p)
        logits = torch.cat([_.unsqueeze(1) for _ in logits], 1).contiguous()
        return logits
