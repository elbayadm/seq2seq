"""
Defining the model's parameters and the colored logging
"""

import os
import os.path as osp
import logging
import configargparse
from .colorstreamhandler import ColorStreamHandler


def create_logger(log_file=None, debug=True):
    """
    Initialize global logger and return it.
    log_file: log to this file, besides console output
    return: created logger
    TODO: optimize the use of the levels in your logging
    """
    logging.basicConfig(level=5,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.root.handlers = []
    if debug:
        chosen_level = 5
    else:
        chosen_level = logging.INFO
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
        # cerate file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(chosen_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Colored stream handler
    sh = ColorStreamHandler()
    sh.setLevel(chosen_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def add_loss_params(parser):
    # Deprectaed  Importance sampling:
    parser.add('--bootstrap', type=int,
               default=0, help='use bootstrap/importance sampling loss.')
    parser.add('--bootstrap_score', type=str,
               default="cider", help='Version of Bootstrap loss')

    parser.add('--gt_loss_version', type=str,
               default="ml", help='Separate loss for the gold caps')
    parser.add('--augmented_loss_version', type=str,
               default="ml", help='Separate loss for the augmented caps')
    # // Deprecated
    # Combining lossess
    parser.add('--alter_loss', type=int, default=0, help='Alter between losses at every iteration')
    parser.add('--alter_mode', type=str, default='even-odd', help='How to altern between losses: even-odd, even-odd-epoch, epoch')
    parser.add('--sum_loss', type=int, default=0, help='Sum two different losses')
    parser.add('--beta', type=float,
               default=0.1, help='Scalar used to weight the losses')
    parser.add('--gamma', type=float,
               default=.33, help='Scalar used to weight the losses')
    parser.add('--combine_loss', type=int,
               default=0, help='combine WL with SL')


    # Loss smoothing
    parser.add('--loss_version', type=str,
               default="ml", help='The loss version:\
               ml: cross entropy,\
               word: word smoothing,\
               seq: sentence smoothing')
    # Generic loss parameters:
    parser.add('--normalize_batch', type=int,
               default=1, help='whether to normalize the batch loss via the mask')
    parser.add('--penalize_confidence', type=float,
               default=0, help='if neq 0, penalize the confidence by subsiding this * H(p) to the loss')
    parser.add('--scale_loss', type=float,
               default=0, help='if neq 0, each sentence loss will be scaled by a pre-computed score (cf dataloader)')

    # loss_version == word params
    parser.add('--similarity_matrix', type=str,
               default='data/WMT14/fr_similarities.pkl',
               help='path to the pre-computed similarity matrix between the vocab words')
    parser.add('--use_cooc', type=int, default=0,
               help='Use cooccurrences matrix instead of glove similarities')
    parser.add('--margin_sim', type=float,
               default=0, help='if neq 0 clip the similarities below this')
    parser.add('--limited_vocab_sim', type=int,
               default=0, help='whether or not to restrain to a subset of similarities\
               0 : the full vocabulary,\
               1 : the 5 captions vocabulary')
    parser.add('--promote_rarity', type=float,
               default=0, help='increase the similarity of rare words')
    parser.add('--rarity_matrix', type=str,
               default='data/IWSLT14/promote_rare.matrix',
               help='path to the pre-computed similarity matrix between the vocab words')
    parser.add('--alpha_word', type=float,
               default=0.9, help='Scalar used to weigh the word loss\
               the final loss = alpha * word + (1-alpha) ml')
    parser.add('--tau_word', type=float,
               default=0.005, help='Temperature applied to the words similarities')

    # loss_version == seq params
    parser.add('--lazy_rnn', type=int,
               default=0, help='lazy estimation of the sampled sentences logp')
    parser.add('--mc_samples', type=int,
               default=1, help='Number of MC samples')
    parser.add('--reward', type=str, default='hamming',
               help='rewards at the seuqence level,\
               options: hamming, bleu1:4, cider, tfidf')
    parser.add('--stratify_reward', type=int,
               default=1, help='sample the reward itself, only possible with reward=Hamming, tfidf')
    parser.add('--importance_sampler', type=str,
               default="greedy", help='the method used to sample candidate sequences,\
               options: greedy (the captioning model itself),\
               hamming: startified sampling of haming')

    parser.add('--alpha_sent', type=float,
               default=0.4, help='Scalar used to weight the losses')
    parser.add('--tau_sent', type=float,
               default=0.1, help='Temperature applied to the sentences scores (r)')
    parser.add('--tau_sent_q', type=float,
               default=0.3, help='Temperature applied to the sentences scores (q) if relevant')

    parser.add('--clip_reward', type=float,
               default=1, help='Upper margin for seq reward clipping')

    # CIDEr specific
    parser.add('--cider_df', type=str,
               default='data/coco-train-df.p', help='path to dataset n-grams frequency')

    # Hamming specific
    parser.add('--limited_vocab_sub', type=int,
               default=1, help='Hamming vocab pool, options:\
               0: the full vocab \
               1: in-batch,\
               2: captions of the image')
    # TFIDF specific
    parser.add('--sub_idf', type=int,
               default=0, help='Substitute commnon ngrams')
    parser.add('--ngram_length', type=int,
               default=2, help='ngram length to substitute')


    ### Alpha scheme:
    parser.add('--alpha_increase_every', type=int,
               default=2, help='step width')
    parser.add('--alpha_increase_factor', type=float,
               default=0.1, help='increase factor when step')
    parser.add('--alpha_max', type=float,
               default=0.9, help='increase factor when step')
    parser.add('--alpha_increase_start', type=int,
               default=1, help='increase factor when step')
    parser.add('--alpha_speed', type=float,
               default=20000, help='alpha decreasing speed')
    parser.add('--alpha_strategy', type=str,
               default="constant", help='Increase strategy')

    return parser


def add_generic_params(parser):
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('--modelname', type=str,
               default='model1', help='directory to store checkpointed models')

    parser.add('--model', type=str,
               default="attention", help='vanilla, attention')
    parser.add('--verbose', type=int, default=0,
               help='code verbosity')
    parser.add('--seed', type=int, default=1, help="seed for all randomizer")

    # Running settings
    # Gpu id if required (LIG servers)
    parser.add('--gpu_id', type=int,
               default=0)
    # Data parameters:
    parser.add('--seq_per_img', type=int, default=1)  # irrelevant here
    parser.add('--max_src_length', type=int, default=50)
    parser.add('--max_trg_length', type=int, default=50)

    parser.add('--input_data_src', type=str,
               default="",
               help='data filename, extension h5 & json will be needed')
    parser.add('--input_data_trg', type=str,
               default="",
               help='data filename, extension h5 & json will be needed')

    parser.add('--batch_size', type=int,
               default=80, help='minibatch size')
    # Decoder parameters:
    parser.add('--bidirectional', type=int, default=1)
    parser.add('--decode', type=str, default='greedy')
    parser.add('--rnn_size_src', type=int,
               default=2000, help='size of the rnn in number of hidden nodes in each layer')
    parser.add('--rnn_type_src', type=str,
               default='lstm', help='rnn, gru, or lstm')
    parser.add('--rnn_size_trg', type=int,
               default=2000, help='size of the rnn in number of hidden nodes in each layer')
    parser.add('--rnn_type_trg', type=str,
               default='lstm', help='rnn, gru, or lstm')
    parser.add('--num_layers_src', type=int,
               default=2, help='number of layers in the RNN')
    parser.add('--num_layers_trg', type=int,
               default=1, help='number of layers in the RNN')
    parser.add('--tied_decoder', type=int,
               default=0, help='tie the decoder embedding with its last mapping to the vocab')
    parser.add('--dim_word_src', type=int,
               default=620,
               help='the encoding size of each token in the vocabulary, and the image.')
    parser.add('--dim_word_trg', type=int,
               default=620,
               help='the encoding size of each token in the vocabulary, and the image.')
    parser.add('--input_encoder_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--encoder_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--enc2dec_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--decoder_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--input_decoder_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--attention_dropout', type=float,
               default=0, help='strength of dropout in the Language Model RNN input')
    parser.add('--scale_grad_by_freq', type=int,
               default=0, help='scale gradient of the embedding layers by the word frequency in the minibatch')

    return parser


def add_optim_params(parser):
    # Optimization: General
    parser.add('--max_epochs', type=int,
               default=5, help='number of epochs')
    parser.add('--grad_clip', type=float,
               default=1, help='clip gradients at this value')

    ## RNN optimizer
    parser.add('--optim', type=str,
               default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add('--optim_alpha', type=float,
               default=0.8, help='alpha for adam')
    parser.add('--optim_beta', type=float,
               default=0.999, help='beta used for adam')
    parser.add('--optim_epsilon', type=float,
               default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add('--weight_decay', type=float,
               default=0, help='main optimizer weight decay')

    ## LR and its scheme
    parser.add('--learning_rate', type=float,
               default=2e-4, help='learning rate')
    parser.add('--learning_rate_decay_start', type=int,
               default=0, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add('--lr_patience', type=int,
               default=2, help='Epochs after overfitting before decreasing the lr')
    parser.add('--lr_strategy', type=str,
               default="step", help="between step (automatic decrease) or adaptive to the val loss")
    parser.add('--learning_rate_decay_every', type=int,
               default=1, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add('--learning_rate_decay_rate', type=float,
               default=0.5, help='every how many iterations thereafter to drop LR?(in epoch)')


    # Scheduled sampling parameters:
    parser.add('--scheduled_sampling_start', type=int,
               default=-1, help='at what iteration to start decay gt probability')
    parser.add('--scheduled_sampling_vocab', type=int,
               default=0, help='if 1 limits sampling to the gt vocab')
    parser.add('--scheduled_sampling_speed', type=int,
               default=100, help='ss speed')
    parser.add('--scheduled_sampling_increase_every', type=int,
               default=5, help='every how many iterations thereafter to gt probability')
    parser.add('--scheduled_sampling_increase_prob', type=float,
               default=0.05, help='How much to update the prob')
    parser.add('--scheduled_sampling_max_prob', type=float,
               default=0.25, help='Maximum scheduled sampling prob.')
    parser.add('--scheduled_sampling_strategy', type=str,
               default="step", help='the decay schedule')
    parser.add('--match_pairs', type=int, #FIXME
               default=0, help='match senetences pairs')
    return parser


def add_vae_params(parser):
    # VAE model
    parser.add('--z_size', type=int,
               default=100, help='VAE/CVAE latent variable size')
    parser.add('--z_interm_size', type=int,
               default=1000, help='intermediary layer between the input and the latent')
    parser.add('--vae_nonlin', type=str,
               default="sigmoid", help="Non-linearity applied to the input of the cvae")
    parser.add('--vae_weight', type=float,
               default=0.1, help="weight of the vae loss (recon + kld)")
    parser.add('--kld_weight', type=float,
               default=0, help="weight of the kld loss")
    return parser

def add_eval_params(parser):
    # Evaluation and Checkpointing
    parser.add('--split', type=str, default="val")
    parser.add('--valid_batch_size', type=int,
               default=25, help='minibatch size')
    parser.add('--forbid_unk', type=int,
               default=1, help='Do not generate UNK tokens when evaluating')
    parser.add('--beam_size', type=int,
               default=3, help='used when sample_max = 1, indicates number of beams in beam search')
    parser.add('--val_images_use', type=int,
               default=-1, help='how many images to use when evaluating (-1 = all)')
    parser.add('--language_eval', type=int,
               default=1, help='Evaluate performance scores (CIDEr, Bleu...)')

    return parser


def parse_opt():
    parser = configargparse.ArgParser()
    # USeful with inria clusters
    parser.add('--restart', type=int,
               default=1, help='0: to override the existing model or 1: to pick up the training')
    # When restarting or finetuning
    parser.add('--start_from_best', type=int, default=0,
               help="Whether to start from the best saved model (1) or the from the last checkpoint (0)")
    # Model to finetune (if restart, the same variable is used to refer to the model itself)
    parser.add('--start_from', type=str,
               default=None, help="The directory of the initialization model, must contain model.pth (resp model-best.pth) \
               the optimizer and the pickled infos")
    parser.add('--save_checkpoint_every', type=int,
               default=4000, help='how often to save a model checkpoint (in iterations)?')
    parser.add('--losses_log_every', type=int,
               default=100, help='How often do we snapshot')
    parser.add('--load_best_score', type=int,
               default=1, help='Do we load previous best score when resuming training.')
    parser = add_generic_params(parser)
    parser = add_loss_params(parser)
    parser = add_optim_params(parser)
    parser = add_eval_params(parser)
    args = parser.parse_args()

    # mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not osp.exists(args.eventname):
        os.makedirs(args.eventname)
    if not osp.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    args.logger = create_logger('%s/train.log' % args.modelname)
    return args


def parse_eval_opt():
    parser = configargparse.ArgParser()
    parser = add_generic_params(parser)
    parser = add_loss_params(parser)
    parser = add_optim_params(parser)
    parser = add_eval_params(parser)
    parser.add('--dump_json', type=int, default=1,
               help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add('--output', type=str,
               help='results file name')
    # When restarting or finetuning
    parser.add('--restart', type=int,
               default=1, help='0: to override the existing model or 1: to pick up the training')
    parser.add('--start_from_best', type=int, default=1,
               help="Whether to start from the best saved model (1) or the from the last checkpoint (0)")
    # Model to finetune (if restart, the same variable is used to refer to the model itself)
    parser.add('--start_from', type=str,
               default=None, help="The directory of the initialization model, must contain model.pth (resp model-best.pth) \
               the optimizer and the pickled infos")
    args = parser.parse_args()

    # mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not osp.exists(args.eventname):
        os.makedirs(args.eventname)
    if not osp.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    args.logger = create_logger('%s/eval.log' % args.modelname)
    return args


