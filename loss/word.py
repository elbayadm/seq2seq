import gc
import time
from math import exp
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.preprocessing as skp
from utils import pl
from .utils import get_ml_loss, get_indices_vocab, to_contiguous
NNZ = 801


def sparse_torch(M):
    """Convert Scipy sparse matrix to torch sparse tensor."""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    # why is this a Variable from the get go
    ST = torch.sparse.FloatTensor(indices, values, shape)
    return ST



def normalize_reward(distrib):
    """
    Normalize so that each row sum to one
    """
    if isinstance(distrib, sp.csr_matrix):
        return skp.normalize(distrib, norm='l1', axis=1)
    if isinstance(distrib, np.ndarray):
        l1 = distrib.sum(axis=1)
        distrib /= l1.reshape(len(distrib), 1)
        return distrib
    # Torch Tensors/Variables:
    sums = torch.sum(distrib, dim=1).unsqueeze(1)
    return distrib / sums.repeat(1, distrib.size(1))


def densify(M, value=0):
    """
    Return dense matrix while replacing zero element
    of the sparse matrix with value
    """
    if isinstance(M, sp.csr_matrix):
        M = M.todense()
        if value:
            M[M == 0] = value
    else:
        M = M.to_dense()
    return M


class WordSmoothCriterion(nn.Module):
    """
    Apply word level loss smoothing given a similarity matrix
    the two versions are:
        full : to take into account the whole vocab
        limited: to consider only the ground truth vocab
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.margin_sim = opt.margin_sim
        self.normalize_batch = opt.normalize_batch
        self.use_cooc = opt.use_cooc
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        if self.margin_sim:
            self.logger.warn('Clipping similarities below %.2f' % self.margin_sim)
        self.limited = opt.limited_vocab_sim
        self.alpha = opt.alpha_word
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pl(opt.similarity_matrix)
        self.dense = isinstance(M, np.ndarray)
        if self.dense:
            if not self.use_cooc:
                M = M - 1  # = -D_ij
            if opt.promote_rarity:
                IDF = pl(opt.rarity_matrix)
                M -= self.tau_word * opt.promote_rarity * IDF
                del IDF
            M = M.astype(np.float32)
            M = Variable(torch.from_numpy(M)).cuda()
            self.Sim_Matrix = M
            n, d = self.Sim_Matrix.size()
        else:
            # self.Sim_Matrix = Variable(sparse_torch(M),
                                       # requires_grad=False).cuda()
            self.Sim_Matrix = sparse_torch(M).cuda()
            n, d = self.Sim_Matrix.size()
        print('Sim matrix:', n, 'x', d, ' V=', opt.vocab_size)
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        self.vocab_size = opt.vocab_size

    def log(self):
        self.logger.info("Initialized Word2 loss tau=%.3f, alpha=%.1f" % (self.tau_word, self.alpha))

    def forward(self, logp, target, mask, scores=None):
        if self.dense:
            return self.forward_dense(logp, target, mask, scores)
        else:
            return self.forward_sparse(logp, target, mask, scores)

    def get_submatrix(self, row_select):
        """
        Return a submatrix of a torch.SparseTensor
        """
        start = time.time()
        nr, nc = self.Sim_Matrix.size()
        dvalue = -1
        if self.tau_word:
            dvalue = exp(-1/self.tau_word)
        subT = dvalue * torch.ones(len(row_select), nc).float().cuda()
        rows, cols = self.Sim_Matrix._indices()
        # Selct rows from row_select
        for e, ind in enumerate(row_select):
            ind = int(ind)
            # ind row covers the range 4000*ind :: 4000*(ind+1)
            # slice_index = np.where(rows.cpu().numpy() == ind)
            # slice_index = np.arange(NNZ*ind, NNZ*(ind+1))
            slice_index = torch.arange(NNZ*ind, NNZ*(ind+1)).long().cuda()
            row_values = self.Sim_Matrix._values()[slice_index]
            row_values -= 1
            if self.tau_word:
                row_values = torch.exp(row_values/self.tau_word)
                row_values = torch.exp(torch.mul(row_values, 1/self.tau_word))
            l1 = torch.sum(row_values) + (nc - NNZ) * dvalue
            subT[e][cols[slice_index]] = row_values/l1
        return subT

    def forward_sparse(self, logp, target, mask, scores=None):
        # truncate to the same size
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        binary_mask = mask
        if scores is not None:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(logp, target, binary_mask, scores,
                                norm=self.normalize_batch,
                                penalize=self.penalize_confidence)
        # Get the similarities of the words in the batch (NxL, V)
        logp = to_contiguous(logp).view(-1, logp.size(2))
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        sim = self.get_submatrix(indices.cpu().numpy())
        sim = Variable(sim, requires_grad=False).cuda()
        mask = to_contiguous(mask).view(-1, 1)
        output = - logp * sim
        # output = - logp * sim
        del sim
        gc.collect()
        if self.normalize_batch:
            if torch.sum(mask).data[0] > 0:
                output = torch.sum(output) / torch.sum(binary_mask)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
        else:
            output = torch.sum(output)

        return ml_output, output, {}

    def forward_dense(self, logp, target, mask, scores=None):
        # truncate to the same size
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        binary_mask = mask
        if scores is not None:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(logp, target, binary_mask, scores,
                                norm=self.normalize_batch,
                                penalize=self.penalize_confidence)
        # Get the similarities of the words in the batch (NxL, V)
        logp = to_contiguous(logp).view(-1, logp.size(2))
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        sim = self.Sim_Matrix[indices]
        if self.margin_sim:
            # keep only the similarities larger than the margin
            sim = sim * sim.ge(self.margin_sim).float()
        if self.limited:
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            logp = logp.gather(1, indices_vocab)
        if self.tau_word:
            smooth_target = torch.exp(torch.mul(sim, 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = sim
        del sim
        # Normalize the word reward distribution:
        smooth_target = normalize_reward(smooth_target)
        # Store some stats about the sentences scores:
        scalars = smooth_target.data.cpu().numpy()
        stats = {"word_mean": np.mean(scalars),
                 "word_std": np.std(scalars)}
        # Format
        mask = to_contiguous(mask).view(-1, 1)
        output = - logp * mask.repeat(1, smooth_target.size(1)) * smooth_target
        del smooth_target
        if self.normalize_batch:
            if torch.sum(mask).data[0] > 0:
                output = torch.sum(output) / torch.sum(binary_mask)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
        else:
            output = torch.sum(output)

        return ml_output, output, stats

    def track(self, logp, target, mask, add_dirac=False):
        """
        Return the prediction distribution & the reward distribution
        """
        # truncate to the same size
        N = logp.size(0)
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        if self.dense:
            sim = self.Sim_Matrix[indices]
        else:
            sim = self.Sim_Matrix[indices].to_dense()
            sim = sim - 1
        if self.margin_sim:
            # keep only the similarities larger than the margin
            sim = sim * sim.ge(self.margin_sim).float()
        if self.limited:
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            logp = logp.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(sim, 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = sim
        # Normalize the word reward distribution:
        smooth_target = normalize_reward(smooth_target)
        if add_dirac and not self.limited:
            delta = Variable(torch.eye(self.vocab_size)[indices.cpu()]).cuda()
            smooth_target = torch.mul(smooth_target, self.alpha) + torch.mul(delta, (1 - self.alpha))

        target_d = smooth_target.view(N, seq_length, -1).data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        return logp, target_d


