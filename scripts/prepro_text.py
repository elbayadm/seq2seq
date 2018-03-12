import os
import sys
import pickle
import json
import argparse
import random
from random import seed
import string
import h5py
import numpy as np
from scipy.misc import imread, imresize
sys.path.insert(0, '.')
from utils import pd

def build_vocab(sentences, params):
    """
    Build vocabulary
    Note: I use the perl scripts instead
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
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    # print('sentence length distribution (count, number of words):')
    # sum_len = sum(sent_lengths.values())
    # for i in range(max_len+1):
        # print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len))

    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<BOS>")
    vocab.insert(0, "<EOS>")
    vocab.insert(0, "<PAD>")
    # Dump the statistics for later use:
    pd({"counts": counts,
        "vocab": vocab,
        "bad words": bad_words,
        "lengths": sent_lengths},
       "data/WMT14/wmt14.stats")

    return vocab


def encode_sentences(sentences, params, wtoi):
    """
    encode all sentences into one large array, which will be 1-indexed.
    No special tokens are added, except from the <pad> after the effective length
    """
    max_length = params['max_length']
    lengths = []
    m = len(sentences)
    IL = np.zeros((m, max_length), dtype='uint32')  # <PAD> token is 0
    M = np.zeros((m, max_length), dtype='uint32')  # <PAD> token is 0
    for i, sent in enumerate(sentences):
        lengths.append(len(sent))
        for k, w in enumerate(sent):
            if k < max_length:
                IL[i, k] = wtoi[w] if w in wtoi else wtoi['<UNK>']
                M[i, k] = int(w in wtoi)
        if not i % 1000:
            print('%.2f%%' % (i / len(sentences) * 100))
    assert np.all(np.array(lengths) > 0), 'error: some line has no words'
    return IL, M, lengths


def main_trg(params):
    """
    Main preprocessing
    """
    with open(params['train_trg'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'] - 1)] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['train_trg']))
    # create the vocab
    # vocab = build_vocab(sentences, params)
    vocab = []
    for line in open(params['vocab_trg'], 'r'):
        vocab.append(line.strip())
    print('Length of vocab:', len(vocab))
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<BOS>")
    vocab.insert(0, "<EOS>")
    vocab.insert(0, "<PAD>")
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    # encode captions in large arrays, ready to ship to hdf5 file
    IL_train, Mask_train, Lengths_train = encode_sentences(sentences, params, wtoi)
    with open(params['val_trg'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'] - 1)] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['val_trg']))
    IL_val, Mask_val, Lengths_val = encode_sentences(sentences, params, wtoi)
    with open(params['test_trg'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'] - 1)] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['test_trg']))
    IL_test, Mask_test, Lengths_test = encode_sentences(sentences, params, wtoi)
    # create output h5 file
    f = h5py.File(params['output_h5_trg'], "w")
    f.create_dataset("labels_train", dtype='uint32', data=IL_train)
    f.create_dataset("mask_train", dtype='uint32', data=Mask_train)
    f.create_dataset("lengths_train", dtype='uint32', data=Lengths_train)


    f.create_dataset("labels_val", dtype='uint32', data=IL_val)
    f.create_dataset("mask_val", dtype='uint32', data=Mask_val)
    f.create_dataset("lengths_val", dtype='uint32', data=Lengths_val)


    f.create_dataset("labels_test", dtype='uint32', data=IL_test)
    f.create_dataset("mask_test", dtype='uint32', data=Mask_test)
    f.create_dataset("lengths_test", dtype='uint32', data=Lengths_test)


    print('wrote ', params['output_h5_trg'])
    pickle.dump({'itow': itow, 'params': params},
                open(params['output_info_trg'], 'wb'))


def main_src(params):
    """
    Main preprocessing
    """
    with open(params['train_src'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'])] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['train_src']))
    # create the vocab
    # vocab = build_vocab(sentences, params)
    vocab = []
    for line in open(params['vocab_src'], 'r'):
        vocab.append(line.strip())
    print('Length of vocab:', len(vocab))
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<PAD>")

    itow = {i: w for i, w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w: i for i, w in enumerate(vocab)} # inverse table
    # encode captions in large arrays, ready to ship to hdf5 file
    IL_train_src, _, Lengths_train = encode_sentences(sentences, params, wtoi)
    with open(params['val_src'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'])] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['val_src']))
    IL_val_src, _, Lengths_val = encode_sentences(sentences, params, wtoi)
    with open(params['test_src'], 'r') as f:
        sentences = f.readlines()
        sentences = [sent.strip().split()[:(params['max_length'])] for sent in sentences]
    print("Read %d lines from %s" % (len(sentences), params['test_src']))
    IL_test_src, _, Lengths_test = encode_sentences(sentences, params, wtoi)

    # create output h5 file
    f = h5py.File(params['output_h5_src'], "w")
    f.create_dataset("labels_train", dtype='uint32', data=IL_train_src)
    f.create_dataset("lengths_train", dtype='uint32', data=Lengths_train)

    f.create_dataset("labels_val", dtype='uint32', data=IL_val_src)
    f.create_dataset("lengths_val", dtype='uint32', data=Lengths_val)

    f.create_dataset("labels_test", dtype='uint32', data=IL_test_src)
    f.create_dataset("lengths_test", dtype='uint32', data=Lengths_test)

    print('wrote ', params['output_h5_src'])
    pickle.dump({'itow': itow, 'params': params},
                open(params['output_info_src'], 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--train_src', type=str, default='data/WMT14/train.en')
    parser.add_argument('--val_src', type=str, default='data/WMT14/dev.en')
    parser.add_argument('--test_src', type=str, default='data/WMT14/test.en')

    parser.add_argument('--train_trg', type=str, default='data/WMT14/train.fr')
    parser.add_argument('--val_trg', type=str, default='data/WMT14/dev.fr')
    parser.add_argument('--test_trg', type=str, default='data/WMT14/test.fr')

    parser.add_argument('--vocab_src', type=str, default='data/WMT14/vocab.en')
    parser.add_argument('--vocab_trg', type=str, default='data/WMT14/vocab.fr')

    parser.add_argument('--output_h5_src', type=str, default='data/WMT14/en_src.h5')
    parser.add_argument('--output_info_src', type=str, default='data/WMT14/en_src.pkl')
    parser.add_argument('--output_h5_trg', type=str, default='data/WMT14/fr_trg.h5')
    parser.add_argument('--output_info_trg', type=str, default='data/WMT14/fr_trg.pkl')

    # options
    parser.add_argument('--max_length', default=50, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main_src(params)
    main_trg(params)
