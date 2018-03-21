"""
Build the similarities matrix used in token-level smoothing
from a given embedding dictionary
"""


import sys
from os.path import expanduser
import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '.')
from utils import pl, pd


def build_embed_dict(embed_txt):
    Glove = {}
    with open(embed_txt, 'r') as f:
        for line in f:
            code = line.strip().split()
            if len(code) < 10:
                continue
            else:
                word = code[0]
                print("parsed word:", word)
                g = np.array(code[1:], dtype="float32")
                Glove[word] = g
    return Glove


def get_pairwise_distances(G):
    eps = 1e-6
    print("G shape:", G.shape, len(G))
    for i in range(len(G)):
        if not np.sum(G[i] ** 2):
            print('%d) norm(g) = 0' % i)
            G[i] = eps + G[i]
    Ds = pdist(G, metric='cosine')
    Ds = squareform(Ds)
    As = np.diag(Ds)
    print("(scipy) sum:", np.sum(As),
          "min:", np.min(Ds), np.min(As),
          "max:", np.max(Ds), np.max(As))
    Ds = 1 - Ds / 2  # map to [0,1]  # FIXME useless
    print(Ds.shape, np.min(Ds), np.max(Ds), np.diag(Ds))
    return Ds


def prepare_embeddings_dict(ixtow, source, output, lower=False):
    """
    From a large dictionary of embeddings get that of the training vocab in order
    inputs:
        ixtow : index to word dictionnary of the vocab
        source: dict of the embedding vectors
        output: dumping path
    """
    dim = source[list(source)[0]].shape[0]
    print('Embedding dimension : %d' % dim)
    G = np.zeros((len(ixtow), dim), dtype="float32")
    for i in range(4, len(ixtow)):
        word = ixtow[i]
        if lower:
            if word.lower() in source:
                G[i] = source[word.lower()]
                if not np.sum(G[i] ** 2):
                    raise ValueError("Norm of the embedding null > token %d | word %s" % (i, word))
        else:
            if word in source:
                G[i] = source[word]
                if not np.sum(G[i] ** 2):
                    raise ValueError("Norm of the embedding null > token %d | word %s" % (i, word))
    pd(G, output)
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_txt', type=str,
                        default="%s/work/GloVe-1.2/iwslt14en/vectors.w15.d300.txt" % expanduser('~'),
                        help='Path to the txt file of the word embeddings')
    parser.add_argument('--data', '-d', type=str, default='IWSLT14',
                        help='Data directory to dump the requested files')
    parser.add_argument('--trg_lang', '-l', type=str, default='en',
                        help='target language')
    parser.add_argument('--embedding', '-e', type=str, default='glove_w15d300',
                        help='type of word embedding')
    parser.add_argument('--save_embed_dict', type=str,
                        help='Path to dump the embedding dict')
    parser.add_argument('--save_embed_matrix', type=str,
                        help='Path to dump the embedding matrix')
    parser.add_argument('--embed_dict', type=str, default='',
                        help='Alternatively give a dictionnary of the coco words embeddings')
    parser.add_argument('--data_info', type=str,
                        help='path of the preprocessing infos file to retrieve ix_to_word')
    parser.add_argument('--data_stats', type=str,
                        help='Path to load coco statistics')
    parser.add_argument('--save_sim', type=str,
                        help='Path to dump the similaritiy matrix')
    parser.add_argument('--save_rarity', type=str,
                        help='path to dump the _rarity_ matrix')
    parser.add_argument('--create_rare_matrix', action='store_true',
                        help='create the rarity matrix for WORSxIDF')

    args = parser.parse_args()
    # define additional params:
    args.save_embed_matrix = "data/%s/%s.embed" % (args.data, args.embedding)
    args.save_embed_dict = "data/%s/%s.dict" % (args.data, args.embedding)
    args.data_info = 'data/%s/%s_trg.infos' % (args.data, args.trg_lang)
    args.data_stats = 'data/%s/vocab.%s.stats' % (args.data, args.trg_lang)
    args.save_sim = 'data/%s/%s.sim' % (args.data, args.embedding)
    args.save_rarity = 'data/%s/promote_rare.matrix' % (args.data)


    if len(args.embed_dict):
        E = pl(args.embed_dict)
    else:
        E = build_embed_dict(args.embed_txt)
        if len(args.save_embed_dict):
            # save for any eventual ulterior usage
            pd(E, args.save_embed_dict)

    ixtow = pl(args.data_info)['itow']
    print("Preparing Glove embeddings matrix")
    embeddings = prepare_embeddings_dict(ixtow, E,
                                         output=args.save_embed_matrix)
    print("Preparing similarities matrix")
    sim = get_pairwise_distances(embeddings)
    print('Saiving the similarity matrix into ', args.save_sim)
    pd(sim.astype(np.float32), args.save_sim)

    if args.create_rare_matrix:
        # Rarity matrix:
        stats = pl(args.data_stats)
        counts = stats['counts']
        total_sentences = sum(list(stats['lengths'].values()))
        print('Total sentences:', total_sentences)
        total_unk = sum([counts[w] for w in stats['bad words']])
        print('Total UNK:', total_unk)
        print('Special tokens:', ixtow[0], ixtow[1], ixtow[2], ixtow[3])
        freq_words = []
        for i in range(4, len(ixtow)):
            try:
                freq_words.append(counts[ixtow[i]])
            except:
                raise ValueError('Missing word: %s (index %d)' % (ixtow[i], i))
        freq = np.array([total_unk] + [total_unk] +
                        [total_sentences] + [total_sentences] +
                        freq_words)
                        # [counts[ixtow[i]] for i in range(4, len(ixtow))])
        print('Frequencies:', freq.shape,
              'min:', np.min(freq), 'max:', np.max(freq))
        F = freq.reshape(1, -1)
        F1 = np.dot(np.transpose(1/F), F)
        F2 = np.dot(np.transpose(F), 1/F)
        FF = np.minimum(F1, F2)
        del F1, F2, F
        print('FF:', FF.shape,
              'min:', np.min(FF), 'max:', np.max(FF))
        pd(FF.astype(np.float32), args.save_rarity)


