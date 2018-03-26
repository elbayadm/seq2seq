import torch
from torch.autograd import Variable
from .layers.utils import to_var


def tile_ctx(ctx, ctx_mask, idxs):
    return ctx[:, idxs], None if ctx_mask is None else ctx_mask[:, idxs]


def beam_search(model, input_lines_src, src_lengths,
                trg_loader, eval_kwargs,
                avoid_double=False, avoid_unk=False):
    """An efficient GPU implementation for beam-search algorithm.
    Arguments:
        model (Model): A model instance derived from `nn.Module` defining
            a set of methods. See `models/nmt.py`.
        data_loader (DataLoader): A ``DataLoader`` instance returned by the
            ``get_iterator()`` method of your dataset.
        vocab (Vocabulary): Vocabulary dictionary for the decoded language.
        beam_size (int, optional): The size of the beam. (Default: 12)
        avoid_double (bool, optional): Suppresses probability of a token if
            it was already decoded in the previous timestep. (Default: False)
        avoid_unk (bool, optional): Prevents <unk> generation. (Default: False)

    Returns:
        list:
            A list of hypotheses in surface form.
    """

    batch_size = input_lines_src.size(0)
    beam_size = eval_kwargs.get('beam_size', 3)
    max_len = trg_loader.seq_length
    bos = trg_loader.bos
    eos = trg_loader.eos
    unk = trg_loader.unk
    n_vocab = trg_loader.get_vocab_size()
    inf = 1e3
    # Mask to apply to pdxs.view(-1) to fix indices
    nk_mask = torch.arange(0, batch_size * beam_size).long().cuda()
    pdxs_mask = (nk_mask / beam_size) * beam_size
    # Tile indices to use in the loop to expand first dim
    tile = nk_mask / beam_size
    # Encode source modalities
    ctx, ctx_mask = model.encode(input_lines_src, src_lengths)
    # We can fill this to represent the beams in tensor format
    beam = torch.zeros((max_len, batch_size, beam_size)).long().cuda()
    # Get initial decoder state (N*H)
    h_t = model.dec.f_init(ctx, ctx_mask)
    # Initial y_t for <bos> embs: N x emb_dim
    y_t = model.dec.emb(to_var(torch.ones(batch_size).long() *
                               bos,
                               volatile=True))
    log_p, h_t = model.dec.f_next(ctx, y_t, h_t)
    nll, beam[0] = log_p.data.topk(beam_size, sorted=False, largest=False)

    for t in range(1, max_len):
        cur_tokens = beam[t - 1].view(-1)
        fini_idxs = (cur_tokens == eos).nonzero()
        n_fini = fini_idxs.numel()
        if n_fini == batch_size * beam_size:
            break
        # Fetch embs for the next iteration (N*K, E)
        y_t = model.dec.emb(to_var(cur_tokens, volatile=True))
        # Get log_probs and new RNN states (log_p, N*K, V)
        ctx, ctx_mask = tile_ctx(ctx, ctx_mask, tile)
        log_p, h_t = model.dec.f_next(ctx, y_t, h_t[tile])
        log_p = log_p.data
        # Suppress probabilities of previous tokens
        if avoid_double:
            log_p.view(-1).index_fill, (
                0, cur_tokens + (nk_mask * n_vocab), inf)

        # Avoid <unk> tokens
        if avoid_unk:
            log_p[:, unk] = inf

        # Favor finished hyps to generate <eos> again
        # Their nll scores will not increase further and they will
        # always be kept in the beam.
        if n_fini > 0:
            fidxs = fini_idxs[:, 0]
            log_p.index_fill_(0, fidxs, inf)
            log_p.view(-1).index_fill_(
                0, fidxs * n_vocab + eos, 0)

        # Expand to 3D, cross-sum scores and reduce back to 2D
        nll = (nll.unsqueeze(2) + log_p.view(batch_size,
                                             beam_size,
                                             -1)).view(batch_size, -1)
        # Reduce (N, K*V) to k-best
        nll, idxs = nll.topk(beam_size, sorted=False, largest=False)
        # previous indices into the beam and current token indices
        pdxs = idxs / n_vocab
        # Insert current tokens
        beam[t] = idxs % n_vocab
        # Permute all hypothesis history according to new order
        beam[:t] = beam[:t].gather(2, pdxs.repeat(t, 1, 1))
        # Compute correct previous indices
        # Mask is needed since we're in flattened regime
        tile = pdxs.view(-1) + pdxs_mask
    # Put an explicit <eos> to make idxs_to_sent happy
    beam[max_len - 1] = eos
    # Find lengths by summing tokens not in (pad,bos,eos)
    lens = (beam.transpose(0, 2) > 2).sum(-1).t().float().clamp(min=1)
    # Normalize scores by length
    nll /= lens.float()
    top_hyps = nll.topk(1, sorted=False, largest=False)[1].squeeze(1)

    # Get best hyp for each sample in the batch
    hyps = beam[:, range(batch_size), top_hyps].cpu().numpy().T
    # results.extend(vocab.list_of_idxs_to_sents(hyps))
    return hyps
