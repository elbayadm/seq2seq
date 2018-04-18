import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .beam_search import Beam
from .seq2seq import Seq2Seq
from .lstm import LSTMAttentionDot
_BOS = 3
_EOS = 2
_UNK = 1
_PAD = 0


class Attention(Seq2Seq):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        """Initialize model."""
        super(Attention, self).__init__(opt, src_vocab_size, trg_vocab_size)
        self.attention_mode = 'dot'
        self.max_trg_length = opt.max_trg_length
        self.rnn_type_src = opt.rnn_type_src.upper()
        self.encoder = getattr(nn, self.rnn_type_src)(self.src_emb_dim,
                                                      self.src_hidden_dim,
                                                      self.nlayers_src,
                                                      bidirectional=self.bidirectional,
                                                      batch_first=True,
                                                      dropout=self.encoder_dropout)

        # self.encoder = nn.DataParallel(self.encoder)
        self.decoder = LSTMAttentionDot(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True,
            dropout=self.attention_dropout
        )

        self.init_weights()

    def get_decoder_init_state(self, input_src, src_lengths):
        """
        Returns:
            src_code : the source code
            state_decoder: mapping of the source's last state
        """
        src_emb = self.input_encoder_dropout(self.src_embedding(input_src))
        if self.pack_seq:
            src_emb = pack_padded_sequence(src_emb,
                                           src_lengths,
                                           batch_first=True)

        state_encoder = self.get_init_state(input_src)
        # h0_encoder, c0_encoder = self.get_init_state(input_src)
        # src_code, (src_h_t, src_c_t) = self.encoder(src_emb, state_encoder)
        src_code, state_encoder = self.encoder(src_emb, state_encoder)
        if self.pack_seq:
            src_code, _ = pad_packed_sequence(src_code,
                                              batch_first=True)  # restore
        if self.bidirectional:
            if self.rnn_type_src == "LSTM":
                h_t = torch.cat((state_encoder[0][-1],
                                 state_encoder[0][-2]), 1)
                c_t = torch.cat((state_encoder[1][-1],
                                 state_encoder[1][-2]), 1)
                state_decoder = (h_t, c_t)

            elif self.rnn_type_src == "GRU":
                h_t = torch.cat((state_encoder[-1],
                                 state_encoder[-2]), 1)
                state_decoder = (h_t)

        else:
            if self.rnn_type_src == "LSTM":
                h_t = state_encoder[0][-1]
                c_t = state_encoder[1][-1]
            elif self.rnn_type_src == "GRU":
                h_t = state_encoder[0][-1]

        h_t = nn.Tanh()(self.enc2dec_dropout(self.encoder2decoder(h_t)))
        state = (h_t, )
        if self.rnn_type_src == "LSTM":
            state = (h_t, c_t)
        # FIXME check if tanh is the best choice
        return src_code, state

    def forward_decoder(self, src_h, state,
                        input_trg, trg_lengths,
                        ctx_mask=None):
        trg_emb = self.input_decoder_dropout(self.trg_embedding(input_trg))
        # trg_emb = pack_padded_sequence(trg_emb,
                                       # trg_lengths,
                                       # batch_first=True)
        ctx = src_h
        trg_h, (_, _) = self.decoder(
            trg_emb,
            state,
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        decoder_logit = F.log_softmax(self.decoder2vocab(self.decoder_dropout(trg_h_reshape)),
                                      dim=1)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def forward(self, input_src, src_lengths, input_trg, trg_lengths, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_code, state = self.get_decoder_init_state(input_src, src_lengths)
        decoder_logit = self.forward_decoder(src_code, state,
                                             input_trg,
                                             trg_lengths,
                                             ctx_mask)
        return decoder_logit

    def sample_beam(self, input_src, src_lengths, ctx_mask=None, opt={}):
        beam_size = opt.get('beam_size', 3)
        batch_size = input_src.size(0)
        # encode the source
        src_code, state = self.get_decoder_init_state(input_src, src_lengths)
        ctx = src_code
        batch_size = ctx.size(1)
        # Expand tensors for each beam.
        context = Variable(ctx.data.repeat(1, beam_size, 1))
        dec_states = [Variable(state[0].data.repeat(1, beam_size, 1)),
                      Variable(state[1].data.repeat(1, beam_size, 1))]

        beam = [Beam(beam_size, opt) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        for i in range(self.opt.max_trg_length):
            input = torch.stack([b.get_current_state()
                                 for b in beam if not b.done]
                                ).t().contiguous().view(1, -1)

            # check if I shoul add _BOS
            trg_emb = self.trg_embedding(Variable(input).transpose(1, 0))
            trg_h, trg_state = self.decoder(trg_emb,
                                            (dec_states[0].squeeze(0),
                                             dec_states[1].squeeze(0)),
                                            context)

            dec_states = [_.unsqueeze(0) for _ in trg_state]
            if i == 4:
                print('post decoding:', dec_states[0].size(), dec_states[1].size())
            # (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = dec_states[0].squeeze(1)
            if i == 4:
                print('dec_out:', dec_out.size())
                print('trg_h:', trg_h.size())
            out = F.softmax(self.decoder2vocab(dec_out),
                            dim=1).unsqueeze(0)

            word_lk = out.view(beam_size,
                               remaining_sents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    # self.model.decoder.hidden_size
                    self.trg_hidden_dim
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        # Wrap up
        allHyp, allScores = [], []
        n_best = 1
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            # hyps = list(zip(*[beam[b].get_hyp(k) for k in ks[:n_best]]))
            hyps = beam[b].get_hyp(ks[0])
            allHyp += [hyps]
        return allHyp, allScores

    def sample(self, input_src, src_lengths, ctx_mask=None, opt={}):
        beam_size = opt.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(input_src, src_lengths, ctx_mask, opt)
        batch_size = input_src.size(0)
        BOS = opt.get('BOS', _BOS)
        EOS = opt.get('EOS', _EOS)

        seq = []
        src_h, state_0 = self.get_decoder_init_state(input_src, src_lengths)
        ctx = src_h
        for t in range(self.opt.max_trg_length):
            if t == 0:
                input_trg = Variable(torch.LongTensor([[BOS] for i in range(batch_size)])).cuda()
                state = state_0
            trg_emb = self.trg_embedding(input_trg)
            trg_h, state = self.decoder(
                trg_emb,
                state,
                ctx,
                ctx_mask
            )
            trg_h_reshape = trg_h.contiguous().view(
                trg_h.size()[0] * trg_h.size()[1],
                trg_h.size()[2]
            )
            decoder_logit = F.log_softmax(self.decoder2vocab(trg_h_reshape),
                                          dim=1)[:, 1:]  # remove the padding pred
            np_logits = decoder_logit.data.cpu().numpy()
            decoder_argmax = 1 + np_logits.argmax(axis=-1)
            if t:
                scores += np_logits[:, decoder_argmax - 1]
            else:
                scores = np_logits[:, decoder_argmax - 1]
            next_preds = Variable(torch.from_numpy(decoder_argmax).view(-1, 1)).cuda()
            seq.append(next_preds)
            input_trg = next_preds
            if t >= 2:
                # stop when all finished
                unfinished = torch.add(torch.mul((input_trg == EOS).type_as(decoder_logit), -1), 1)
                if unfinished.sum().data[0] == 0:
                    break

        seq = torch.cat(seq, 1).data.cpu().numpy()
        return seq, scores

