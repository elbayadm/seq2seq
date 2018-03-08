import numpy as np
import torch
import torch.nn as nn
from .utils import to_contiguous, get_ml_loss


class MLCriterion(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.normalize_batch = opt.normalize_batch
        self.penalize_confidence = opt.penalize_confidence

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target, mask, scores=None):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        output = get_ml_loss(logp, target, mask, scores,
                             norm=self.normalize_batch,
                             penalize=self.penalize_confidence)
        return output, output, None

    def track(self, logp, target, mask, add_dirac=False):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        """
        # truncate to the same size
        N = logp.size(0)
        seq_length = logp.size(1)
        target = target[:, :seq_length].data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        target_d = np.zeros_like(logp)
        rows = np.arange(N).reshape(-1, 1).repeat(seq_length, axis=1)
        cols = np.arange(seq_length).reshape(1, -1).repeat(N, axis=0)
        target_d[rows, cols, target] = 1
        return logp, target_d


