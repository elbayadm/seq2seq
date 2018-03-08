import os
import pickle
import json
import argparse
import random
from random import seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize


def vocab_stats(sentences):
    """
    Build vocabulary
    """
    # count up the number of words
    counts = {}
    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for txt in sentences:
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
        for w in txt:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    vocab = [w for (c, w) in cw[:30000]]
    bad_words = [w for (c, w) in cw[30000:]]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words),
                                                   len(counts),
                                                   len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count,
                                              total_words,
                                              bad_count*100.0/total_words))
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    # for i in range(max_len+1):
        # print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0),
                                    # sent_lengths.get(i, 0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.insert(0, '<EOS>')
        vocab.insert(0, '<BOS>')
        vocab.insert(0, '<UNK>')

    return vocab, cw, sent_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--corpus', type=str, default='data/WMT14/train.fr')
    # options
    parser.add_argument('--max_length',
                        default=50,
                        type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    sentences = []
    for line in open(params['corpus'], encoding="utf8"):
        sentences.append(line.strip().split()[:(params['max_length'])])
    print("Read %d lines from %s" % (len(sentences), params['corpus']))
    vocab, counts, lengths = vocab_stats(sentences)
    data = {"vocab": vocab,
            "counts": counts,
            "lengths": lengths}
    pickle.dump(data, open('data/WMT14/vocab_fr_stats.pkl', "wb"))


