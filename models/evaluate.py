# -*- coding: utf-8 -*-

"""Evaluation utils."""
import os
from collections import Counter
import math
import time
import subprocess
import numpy as np
import torch
from torch.autograd import Variable
from utils import decode_sequence
import utils.logging as lg


# ESKE
def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a list of translation hypotheses and references.
    With the default settings, this computes the exact same score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n. `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((order,))
    correct = np.zeros((order,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        hyp_length += len(hyp)
        ref_length += len(ref)

        for i in range(order):
            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))
            ref_ngrams = Counter(zip(*[ref[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    def divide(x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.true_divide(x, y)
            z[~ np.isfinite(z)] = 0
        return z

    scores = divide(correct, total)

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length)) if hyp_length > 0 else 0.0
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""

    def bleu_stats(hypothesis, reference):
        """Compute statistics for BLEU."""
        stats = []
        stats.append(len(hypothesis))
        stats.append(len(reference))
        for n in range(1, 5):
            s_ngrams = Counter(
                [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
            )
            r_ngrams = Counter(
                [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
            )
            stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
            stats.append(max([len(hypothesis) + 1 - n, 0]))
        return stats

    def bleu(stats):
        """Compute BLEU given n-gram statistics."""
        if len([x for x in stats if x == 0]) > 0:
            return 0
        (c, r) = stats[:2]
        log_bleu_prec = sum(
            [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
        ) / 4.
        return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", 'multi-bleu.perl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()


def model_perplexity(model, src_loader, trg_loader, split="val", logger=None):
    """Compute model perplexity."""
    # Make sure to be in evaluation mode
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    loss_evals = 0
    while True:
        # get batch
        data_src = src_loader.get_src_batch(split)
        input_lines_src = data_src['labels']
        input_lines_src = Variable(torch.from_numpy(input_lines_src),
                                   requires_grad=False).cuda()

        data_trg = trg_loader.get_trg_batch(split)
        tmp = [data_trg['labels'], data_trg['out_labels'], data_trg['mask']]
        input_lines_trg, output_lines_trg, mask = [Variable(torch.from_numpy(_),
                                                            requires_grad=False).cuda()
                                                   for _ in tmp]

        n = n + src_loader.batch_size
        decoder_logit = model(input_lines_src, input_lines_trg)
        ml_loss, loss, stats = model.crit(decoder_logit, output_lines_trg, mask)
        loss_sum = loss_sum + loss.data[0]
        loss_evals = loss_evals + 1

        ix1 = data_src['bounds']['it_max']
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    return loss_sum / loss_evals


def evaluate_model(model, src_loader, trg_loader, logger, eval_kwargs):
    """Evaluate model."""
    preds = []
    ground_truths = []
    batch_size = eval_kwargs.get('batch_size', 1)
    split = eval_kwargs.get('split', 'val')
    verbose = eval_kwargs.get('verbose', 0)
    # Make sure to be in evaluation mode
    model.eval()
    src_loader.reset_iterator(split)
    trg_loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    while True:
        # get batch
        data_src = src_loader.get_src_batch(split, batch_size)
        input_lines_src = data_src['labels']
        input_lines_src = Variable(torch.from_numpy(input_lines_src),
                                   requires_grad=False).cuda()

        data_trg = trg_loader.get_trg_batch(split, batch_size)
        tmp = [data_trg['labels'], data_trg['out_labels'], data_trg['mask']]
        input_lines_trg_gold, output_lines_trg_gold, mask = [Variable(torch.from_numpy(_),
                                                                     requires_grad=False).cuda()
                                                            for _ in tmp]

        n += batch_size
        # decoder_logit = model(input_lines_src, input_lines_trg_gold)
        # if model.opt.sample_reward:
            # ml_loss, loss, stats = model.crit(model, input_lines_src, input_lines_trg_gold,
                                              # output_lines_trg_gold, mask)
        # else:
            # ml_loss, loss, stats = model.crit(decoder_logit, output_lines_trg_gold, mask)

        ml_loss, loss, _ = model.step(input_lines_src,
                                      input_lines_trg_gold,
                                      output_lines_trg_gold,
                                      mask)
        loss_sum += loss.data[0]
        ml_loss_sum += ml_loss.data[0]
        loss_evals = loss_evals + 1
        # Initialize target with <BOS> for every sentence Index = 2
        # print('Sampling sentence')
        # print('GPU:', os.environ['CUDA_VISIBLE_DEVICES'])
        start = time.time()
        # Decode a minibatch greedily __TODO__ add beam search decoding
        batch_preds, _ = model.sample(input_lines_src, opt=eval_kwargs)
        if isinstance(batch_preds, list):
            # wiht beam size unpadded preds
            sent_preds = [decode_sequence(trg_loader.get_vocab(),
                                          np.array(pred).reshape(1, -1))[0]
                          for pred in batch_preds]
        else:
            # decode
            sent_preds = decode_sequence(trg_loader.get_vocab(), batch_preds)
        # Do the same for gold sentences
        sent_source = decode_sequence(src_loader.get_vocab(), input_lines_src.data.cpu().numpy())
        sent_gold = decode_sequence(trg_loader.get_vocab(), output_lines_trg_gold.data.cpu().numpy())
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (sl, l, gl) in zip(sent_source, sent_preds, sent_gold):
            preds.append(l)
            ground_truths.append(gl)
            if verb:
                lg.print_sampled(sl, gl, l)
        ix1 = data_src['bounds']['it_max']
        if data_src['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    # print('Predictions lenght:', len(preds), len(ground_truths))
    # assert(len(preds) == trg_loader.h5_file['labels_val'].shape[0])
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    return preds, ml_loss_sum / loss_evals, loss_sum / loss_evals, bleu_moses


def score_trads(preds, trg_loader,  eval_kwargs):
    split = eval_kwargs.get('split', 'val')
    batch_size = eval_kwargs.get('batch_size', 80)
    verbose = eval_kwargs.get('verbose', 0)
    ground_truths = []
    trg_loader.reset_iterator(split)
    n = 0
    while True:
        # get batch
        data_trg = trg_loader.get_trg_batch(split, batch_size)
        output_lines_trg_gold = data_trg['out_labels']
        n += batch_size
        # Decode a minibatch greedily __TODO__ add beam search decoding
        # Do the same for gold sentences
        sent_gold = decode_sequence(trg_loader.get_vocab(), output_lines_trg_gold)
        if not verbose:
            verb = not (n % 1000)
        else:
            verb = verbose
        for (l, gl) in zip(preds, sent_gold):
            ground_truths.append(gl)
            if verb:
                lg.print_sampled("", gl, l)
        ix1 = data_trg['bounds']['it_max']
        if data_trg['bounds']['wrapped']:
            break
        if n >= ix1:
            print('Evaluated the required samples (%s)' % n)
            break
    bleu_moses, _ = corpus_bleu(preds, ground_truths)
    scores = {'Bleu': bleu_moses}
    return scores


