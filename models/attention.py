import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .beam_search import Beam
from .seq2seq import Seq2Seq
from .lstm import LSTMAttentionDot
_BOS = 2
_EOS = 1


class Attention(Seq2Seq):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        """Initialize model."""
        super(Attention, self).__init__(opt, src_vocab_size, trg_vocab_size)
        self.attention_mode = 'dot'

        self.encoder = nn.LSTM(
            self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True
        )

        self.init_weights()

    def get_decoder_init_state(self, input_src):
        """
        Returns:
            src_h : the source code
            h_t: mapping of the source code src_h_t=T
            c_t : the source final context src_c_t=T

        """
        src_emb = self.src_embedding(input_src)
        # print('Embedded into:', src_emb.size())
        h0_encoder, c0_encoder = self.get_init_state(input_src)
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (h0_encoder, c0_encoder)
        )
        if self.bidirectional:
            # print("Concat:", src_h_t[-1].size(), src_h_t[-2].size())
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        # print('passing ', h_t.size())
        h_t = nn.Tanh()(self.encoder2decoder(h_t))  # FIXME should i drop it
        # return h_t, src_h, c_t  # FIXME
        return src_h, (h_t, c_t)


    def forward_decoder(self, decoder_init_state,
                        src_h, c_t, input_trg,
                        ctx_mask=None):
        trg_emb = self.trg_embedding(input_trg)
        ctx = src_h.transpose(0, 1)
        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_init_state, c_t),
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        # print("Softmax >>>")
        # print('in:', self.trg_hidden_dim, self.trg_vocab_size, 'recieved:', trg_h_reshape.size())
        decoder_logit = F.log_softmax(self.decoder2vocab(trg_h_reshape))
        # print('output:', decoder_logit.size())
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_h, (h_t, c_t) = self.get_decoder_init_state(input_src)
        decoder_logit = self.forward_decoder(h_t,
                                             src_h, c_t,
                                             input_trg,
                                             ctx_mask)
        return decoder_logit

    def sample_beam(self, input_src, ctx_mask=None, opt={}):
        beam_size = opt.get('beam_size', 3)
        batch_size = input_src.size(0)
        # encode the source
        src_h, (h_0, c_0) = self.get_decoder_init_state(input_src)
        # context_h, (context_h_t, context_c_t) = self.get_decoder_init_state(input_src)
        # Switch to L, N i.e. sequence first
        ctx = src_h.transpose(0, 1)
        batch_size = ctx.size(1)
        # Expand tensors for each beam.
        context = Variable(ctx.data.repeat(1, beam_size, 1))
        dec_states = [Variable(h_0.data.repeat(1, beam_size, 1)),
                      Variable(c_0.data.repeat(1, beam_size, 1))]

        beam = [Beam(beam_size, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        for i in range(self.opt.max_trg_length):
            input = torch.stack([b.get_current_state()
                                 for b in beam if not b.done]
                                ).t().contiguous().view(1, -1)

            # check if I shoul add _BOS
            trg_emb = self.trg_embedding(Variable(input).transpose(1, 0))
            trg_h, (trg_h_t, trg_c_t) = self.decoder(trg_emb,
                                                     (dec_states[0].squeeze(0),
                                                      dec_states[1].squeeze(0)),
                                                     context)

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            out = F.softmax(self.decoder2vocab(dec_out)).unsqueeze(0)

            word_lk = out.view( beam_size,
                               remaining_sents,
                               -1).transpose(0, 1).contiguous()

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

    def sample(self, input_src, ctx_mask=None, opt={}):
        beam_size = opt.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(input_src, ctx_mask, opt)
        batch_size = input_src.size(0)
        seq = []
        # print('input_src dim:', input_src.size())
        src_h, (h_0, c_0) = self.get_decoder_init_state(input_src)
        ctx = src_h.transpose(0, 1)

        for t in range(self.opt.max_trg_length):
            if t == 0:
                input_trg = Variable(torch.LongTensor([[_BOS] for i in range(batch_size)])).cuda()
                # print("input dim:", input_trg.size())
                h_t = h_0
                c_t = c_0
            trg_emb = self.trg_embedding(input_trg)
            # print('state:', h_t.size(), c_t.size(), "trgemb:", trg_emb.size())

            trg_h, (h_t, c_t) = self.decoder(
                trg_emb,
                (h_t, c_t),
                ctx,
                ctx_mask
            )

            trg_h_reshape = trg_h.contiguous().view(
                trg_h.size()[0] * trg_h.size()[1],
                trg_h.size()[2]
            )
            decoder_logit = F.log_softmax(self.decoder2vocab(trg_h_reshape))[:, 1:]  # remove the padding pred
            # print(decoder_logit)
            np_logits = decoder_logit.data.cpu().numpy()
            decoder_argmax = 1 + np_logits.argmax(axis=-1)
            if t:
                scores += np_logits[:, decoder_argmax - 1]
            else:
                scores = np_logits[:, decoder_argmax - 1]
            # print('argmax:', decoder_argmax)
            next_preds = Variable(torch.from_numpy(decoder_argmax).view(-1, 1)).cuda()
            # print("Next tokens:", next_preds.size())
            seq.append(next_preds)
            input_trg = next_preds
            if t >= 2:
                # stop when all finished
                unfinished = torch.add(torch.mul((input_trg == _EOS).type_as(decoder_logit), -1), 1)
                # print('Unfinished:', unfinished)
                if unfinished.sum().data[0] == 0:
                    break
                # print("input_trg:", input_trg)

        seq = torch.cat(seq, 1).data.cpu().numpy()
        print('scores:', scores)
        return seq, scores

