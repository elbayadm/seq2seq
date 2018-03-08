import random
import math
from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .word import WordSmoothCriterion
from .utils import decode_sequence, get_ml_loss
from .samplers import init_sampler


class RewardSampler(nn.Module):
    """
    Sampling the sentences wtr the reward distribution
    instead of the captionig model itself
    """
    def __init__(self, opt, vocab):
        super(RewardSampler, self).__init__()
        self.logger = opt.logger
        self.penalize_confidence = opt.penalize_confidence
        self.lazy_rnn = opt.lazy_rnn
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        self.combine_loss = opt.combine_loss
        self.vocab = vocab
        self.verbose = opt.verbose
        self.mc_samples = opt.mc_samples
        if self.combine_loss:
            self.loss_sampled = WordSmoothCriterion(opt)
            # self.loss_gt = WordSmoothCriterion(opt)
            self.loss_gt = self.loss_sampled  # for now it's the same
        self.sampler = init_sampler(opt.reward.lower(), opt)

    def log(self):
        self.logger.info('RewardSampler (stratified sampling), r=%s' % self.sampler.version)
        if self.combine_loss:
            self.logger.info('GT loss:')
            self.loss_gt.log()
            self.logger.info('Sampled loss:')
            self.loss_sampled.log()

    def forward(self, model,
                input_lines_src,
                input_lines_trg,
                output_lines_trg,
                mask, scores=None):
        ilabels = input_lines_trg
        olabels = output_lines_trg
        # truncate
        # logp = model.forward_decoder(decoder_init_state,
                                     # src_h, src_c,
                                     # ilabels)

        logp = model.forward(input_lines_src, ilabels)
        # Remove BOS token
        seq_length = logp.size(1)
        target = olabels[:, :seq_length]
        del olabels, ilabels
        mask = mask[:, :seq_length]
        if scores is not None:
            print('scaling the masks')
            # FIXME see to it that i do not normalize with scaled masks
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)

        # GT loss
        loss_gt, stats = self.batch_loss_lazy(logp, target, mask, scores)
        # Sampling loss
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for mci in range(MC):
            ipreds_matrix, opreds_matrix, _, stats = self.sampler.nmt_sample(logp, target)
            if self.lazy_rnn:
                mc_output, stats_sampled = self.batch_loss_lazy(logp, opreds_matrix, mask, scores)
            else:
                # Forward the sampled captions
                mc_output, stats_sampled = self.batch_loss(model,
                                                           input_lines_src,
                                                           # decoder_init_state,
                                                           # src_h, src_c,
                                                           ipreds_matrix,
                                                           opreds_matrix,
                                                           mask, scores)
            if not mci:
                output = mc_output
            else:
                output += mc_output
        output /= MC
        return loss_gt, output, stats

    def batch_loss(self, model,
                   input_lines_src,
                   # decoder_init_state, src_h, src_c,
                   ipreds_matrix, opreds_matrix, mask, scores):
        """
        forward the new sampled labels and return the loss
        """
        # logp = model.forward_decoder(decoder_init_state,
                                     # src_h, src_c,
                                     # ipreds_matrix)
        logp = model.forward(input_lines_src, ipreds_matrix)
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              opreds_matrix,
                                              mask,
                                              scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)
        else:
            ml = get_ml_loss(logp,
                             opreds_matrix,
                             mask,
                             scores,
                             penalize=self.penalize_confidence)
            loss = ml
            stats = None
        return loss, stats

    def batch_loss_lazy(self, logp, target, mask, scores):
        """
        Evaluate the loss of the new labels given the gt logits
        """
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp, target, mask, scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)

        else:
            ml = get_ml_loss(logp, target, mask, scores)
            loss = ml
            stats = None
        return loss, stats


