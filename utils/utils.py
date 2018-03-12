import os
import os.path as osp
import sys
from collections import Counter
import math
import pickle
from scipy.special import binom
import numpy as np
from scipy.spatial.distance import hamming

_BOS = 2
_EOS = 1


def pl(path):
    return pickle.load(open(path, 'rb'),
                       encoding='iso-8859-1')


def pd(obj, path):
    pickle.dump(obj, open(path, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


def to_contiguous(tensor):
    """
    Convert tensor if not contiguous
    """
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()



def decode_sequence(ix_to_word, seq):
    """
    Decode sequence into natural language
    Input: seq, N*D numpy array, with element 0 .. vocab_size.
    """
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = []
        for j in range(D):
            ix = seq[i, j]
            if ix > 0 and not ix == _EOS:
                if ix == _BOS:
                    continue
                else:
                    txt.append(ix_to_word[ix])
            else:
                break
        out.append(' '.join(txt))
    return out


def group_similarity(u, refs):
    sims = []
    for v in refs:
        sims.append(1 + np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return np.mean(sims)


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    """
    Compute sentence-level BLEU score between a translation hypothesis and a reference.

    :param hypothesis: list of tokens or token ids
    :param reference: list of tokens or token ids
    :param smoothing: apply smoothing (recommended, especially for short sequences)
    :param order: count n-grams up to this value of n.
    :param kwargs: additional (unused) parameters
    :return: BLEU score (float)
    """
    log_score = 0

    if len(hypothesis) == 0:
        return 0

    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))

        numerator = sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())

        if smoothing:
            numerator += 1
            denominator += 1

        score = numerator / denominator

        if score == 0:
            log_score += float('-inf')
        else:
            log_score += math.log(score) / order

    bp = min(1, math.exp(1 - len(reference) / len(hypothesis)))

    return math.exp(log_score) * bp


def manage_lr(epoch, opt, val_losses):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        opt.logger.error('Updating the lr')
        if opt.lr_strategy == "step":
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate  ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            opt.scale_lr = decay_factor
        elif opt.lr_strategy == "adaptive":
            opt.logger.error('Adaptive mode')
            print("val_losses:", val_losses)
            if len(val_losses) > 2:
                if val_losses[0] > val_losses[1]:
                    opt.lr_wait += 1
                    opt.logger.error('Waiting for more')
                if opt.lr_wait > opt.lr_patience:
                    opt.logger.error('You have plateaued, decreasing the lr')
                    # decrease lr:
                    opt.current_lr = opt.current_lr * opt.learning_rate_decay_rate
                    opt.scale_lr = opt.learning_rate_decay_factor
                    opt.lr_wait = 0
            else:
                opt.current_lr = opt.learning_rate
                opt.scale_lr = 1
    else:
        opt.current_lr = opt.learning_rate
        opt.scale_lr = 1
    return opt


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def scale_lr(optimizer, scale):
    for group in optimizer.param_groups:
        group['lr'] *= scale


def clip_gradient(optimizer, max_norm, norm_type=2):
    max_norm = float(max_norm)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for group in optimizer.param_groups for p in group['params'])
    else:
        total_norm = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    nn = param_norm ** norm_type
                    # print('norm:', nn, p.grad.size())
                    total_norm += nn
                    param_norm ** norm_type
                except:
                    pass
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for group in optimizer.param_groups:
            for p in group['params']:
                try:
                    p.grad.data.mul_(clip_coef)
                except:
                    pass
    return total_norm



