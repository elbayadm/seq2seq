import torch.nn as nn
from .layers import text_encoder, cond_decoder
from .layers.utils import to_var
import loss

class seq2seq(nn.Module):
    """
    A sequence-to-sequence NMT model with attention.
    """
    def __init__(self, opt, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.opt = opt
        self.logger = opt.logger
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        # Encoder
        self.enc = text_encoder(opt, src_vocab_size)
        # Decoder
        self.dec = cond_decoder(opt, trg_vocab_size)
        self.ss_prob = 0.  # scheduled sampling

    def init_weights(self):
        """
        Initialize the model's weights
        """
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal(param.data)

    def encode(self, input, lengths):
        """
        Given source data, return the code to pass to the decoder
        return hs, mask
        """
        return self.enc(input, lengths)

    def forward(self, source, source_lengths, target):
        """
        Return the decoder logits
        """
        # encode source
        ctx, ctx_mask = self.encode(source, source_lengths)
        print('ctx:', ctx.size(), 'mask:', ctx_mask.size())
        # decode in the target language
        logits = self.dec(ctx, ctx_mask, target)
        return logits

    def define_loss(self, trg_loader):
        opt = self.opt
        ver = opt.loss_version.lower()
        if ver == 'ml':
            crit = loss.MLCriterion(opt)
        elif ver == 'word':
            crit = loss.WordSmoothCriterion(opt)
        elif ver == "seq":
            if opt.stratify_reward:
                crit = loss.RewardSampler(opt, trg_loader)
            else:
                crit = loss.ImportanceSampler(opt, trg_loader)
        else:
            raise ValueError('unknown loss mode %s' % ver)
        crit.log()
        self.crit = crit

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            for k in list(saved):
                if 'crit' in k:
                    self.logger.warn('Deleting key %s' % k)
                    del saved[k]
            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            self.load_state_dict(saved)

    def step(self, input_lines_src, src_lengths,
             input_lines_trg, trg_lengths, output_lines_trg, mask):
        opt = self.opt
        if opt.loss_version.lower() == "seq":
            # Avoid re-encoding the sources when sampling other targets
            src_code = self.encode(input_lines_src, src_lengths)
            ml_loss, reward_loss, stats = self.crit(self, src_code,
                                                    input_lines_trg,
                                                    output_lines_trg,
                                                    mask)

        else:
            # init and forward decoder combined
            decoder_logit = self.forward(input_lines_src,
                                         src_lengths,
                                         input_lines_trg)
            ml_loss, reward_loss, stats = self.crit(decoder_logit,
                                                    output_lines_trg,
                                                    mask)

        if opt.loss_version.lower() == "ml":
            final_loss = ml_loss
        else:
            final_loss = self.crit.alpha * reward_loss + (1 - self.crit.alpha) * ml_loss
        return ml_loss, final_loss, stats



