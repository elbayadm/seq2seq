import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import pl
from .utils import get_ml_loss, get_indices_vocab, to_contiguous


def normalize_reward(distrib):
    """
    Normalize so that each row sum to one
    """
    sums = torch.sum(distrib, dim=1).unsqueeze(1)
    return distrib / sums.repeat(1, distrib.size(1))


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
        if not self.use_cooc:
            M = M - 1  # = -D_ij
        if opt.promote_rarity:
            IDF = pl(opt.rarity_matrix)
            M -= self.tau_word * opt.promote_rarity * IDF
            del IDF
        M = M.astype(np.float32)
        n, d = M.shape
        print('Sim matrix:', n, 'x', d, ' V=', opt.vocab_size)
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        self.vocab_size = opt.vocab_size
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M
        del M

    def log(self):
        self.logger.info("Initialized Word2 loss tau=%.3f, alpha=%.1f" % (self.tau_word, self.alpha))

    def forward(self, logp, target, mask, scores=None):
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
        # Normalize the word reward distribution:
        smooth_target = normalize_reward(smooth_target)
        if add_dirac and not self.limited:
            delta = Variable(torch.eye(self.vocab_size)[indices.cpu()]).cuda()
            smooth_target = torch.mul(smooth_target, self.alpha) + torch.mul(delta, (1 - self.alpha))

        target_d = smooth_target.view(N, seq_length, -1).data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        return logp, target_d


