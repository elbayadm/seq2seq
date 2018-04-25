"""Sequence to Sequence parent model."""
import os.path as osp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import loss


class Seq2Seq(nn.Module):
    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        """Initialize model."""
        nn.Module.__init__(self)
        self.opt = opt
        self.logger = opt.logger
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.pack_seq = opt.pack_seq
        self.seq_length = opt.max_trg_length
        self.src_emb_dim = opt.dim_word_src
        self.trg_emb_dim = opt.dim_word_trg
        self.src_hidden_dim = opt.rnn_size_src
        self.trg_hidden_dim = opt.rnn_size_trg
        self.ctx_hidden_dim = opt.rnn_size_src
        self.batch_size = opt.batch_size
        assert self.trg_hidden_dim == self.ctx_hidden_dim, 'Sizes mismatch'  # FIXME

        self.bidirectional = opt.bidirectional
        self.nlayers_src = opt.num_layers_src
        self.nlayers_trg = opt.num_layers_trg
        self.attention_dropout = opt.attention_dropout
        # applied on the outputs of each rnn except the last (c.f. nn.LSTM)
        self.encoder_dropout = opt.encoder_dropout
        self.input_encoder_dropout = nn.Dropout(opt.input_encoder_dropout)
        # applied on the encoder code before feeding it to the decoder
        # applied to the decoder outputs before mapping to the vocab size.
        self.input_decoder_dropout = nn.Dropout(opt.input_decoder_dropout)
        self.num_directions = 2 if self.bidirectional else 1
        self.pad_token_src = 0
        self.pad_token_trg = 0
        self.ss_prob = 0. # scheduled sampling

        self.src_hidden_dim = self.src_hidden_dim // 2 \
            if self.bidirectional else self.src_hidden_dim

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src,
            scale_grad_by_freq=bool(opt.scale_grad_by_freq)

        )
        self.trg_embedding = nn.Embedding(
            self.trg_vocab_size,
            self.trg_emb_dim,
            self.pad_token_trg,
            scale_grad_by_freq=bool(opt.scale_grad_by_freq)

        )
        # print('Enc2Dec dims:', self.src_hidden_dim * self.num_directions, self.trg_hidden_dim)
        self.enc2dec_dropout = nn.Dropout(opt.enc2dec_dropout)
        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim,
                                       self.trg_vocab_size)
        self.decoder_dropout = nn.Dropout(opt.decoder_dropout)

    def init_weights(self):
        """Initialize weights."""
        # initrange = 0.1
        # self.src_embedding.weight.data.uniform_(-initrange, initrange)
        # self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        initdev = 0.01
        self.src_embedding.weight.data.normal_(0.0, initdev)
        self.trg_embedding.weight.data.normal_(0.0, initdev)

        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_init_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0)
        h0_encoder = Variable(torch.zeros(
            self.nlayers_src * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.nlayers_src * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=1)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs

    def define_loss(self, trg_loader):
        opt = self.opt
        ver = opt.loss_version.lower()
        if ver == 'ml':
            crit = loss.MLCriterion(opt)
        elif ver == 'word':
            crit = loss.WordSmoothCriterion(opt)
        elif ver == "seq":
            if opt.stratify_reward:
                crit = loss.RewardSampler(opt, trg_loader)
            else:
                crit = loss.ImportanceSampler(opt, trg_loader)
        else:
            raise ValueError('unknown loss mode %s' % ver)
        crit.log()
        self.crit = crit

    def define_old_loss(self, vocab):
        opt = self.opt
        if opt.sample_cap:
            # Sampling from the captioning model itself
            if 'cider' in opt.loss_version:
                crit = loss.CiderRewardCriterion(opt, vocab)
            elif 'hamming' in opt.loss_version:
                crit = loss.HammingRewardCriterion(opt, vocab)
            elif 'infersent' in opt.loss_version:
                crit = loss.InfersentRewardCriterion(opt, vocab)
            elif 'bleu' in opt.loss_version:
                crit = loss.BleuRewardCriterion(opt, vocab)
            elif opt.loss_version == "word":
                crit = loss.WordSmoothCriterion(opt, vocab)
            elif opt.loss_version == "word2":
                crit = loss.WordSmoothCriterion2(opt)

        elif opt.sample_reward:
            if 'hamming' in opt.loss_version:
                crit = loss.HammingRewardSampler(opt, vocab)
            else:
                raise ValueError('Loss function %s in sample_reward mode unknown' % (opt.loss_version))

        elif opt.bootstrap:
            crit = loss.DataAugmentedCriterion(opt)
        elif opt.combine_caps_losses:
            crit = loss.MultiLanguageModelCriterion(opt.seq_per_img)
        else:
            # The defualt ML
            opt.logger.warn('Using baseline loss criterion')
            crit = loss.LanguageModelCriterion(opt)
        self.crit = crit

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            for k in list(saved):
                if 'crit' in k:
                    self.logger.warn('Deleting key %s' % k)
                    del saved[k]
            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            required_keys = list(self.state_dict())
            for k in required_keys:
                if k not in saved:
                    if "module" in k:
                        ki = k.split(".")
                        ki.remove('module')
                        kk = '.'.join(ki)
                        assert kk in saved
                        saved[k] = saved[kk]
                        del saved[kk]
                    else:
                        # Add module
                        self.logger.warn('Issue the key %s' % k)
            self.load_state_dict(saved)

    def step(self, input_lines_src, src_lengths,
             input_lines_trg, trg_lengths, output_lines_trg, mask):
        opt = self.opt
        if opt.loss_version.lower() == "seq":
            # Avoid re-encoding the sources when sampling other targets
            src_code, state = self.get_decoder_init_state(input_lines_src, src_lengths)
            ml_loss, reward_loss, stats = self.crit(self, src_code, state,
                                                    input_lines_trg,
                                                    trg_lengths,
                                                    output_lines_trg,
                                                    mask)

        else:
            # init and forward decoder combined
            decoder_logit = self.forward(input_lines_src, src_lengths, input_lines_trg, trg_lengths)
            ml_loss, reward_loss, stats = self.crit(decoder_logit,
                                                    output_lines_trg,
                                                    mask)

        if opt.loss_version.lower() == "ml":
            final_loss = ml_loss
        else:
            final_loss = self.crit.alpha * reward_loss + (1 - self.crit.alpha) * ml_loss
        return ml_loss, final_loss, stats



