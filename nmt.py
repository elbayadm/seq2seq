# -*- coding: utf-8 -*-
"""
Main training loop
"""

import sys
import os
import gc
import time
import random
import numpy as np
from options import parse_opt, get_gpu_memory


def train(opt):
    # setup gpu
    try:
        import subprocess
        # gpu_id = subproces.check_output('source gpu_setVisibleDevices.sh', shell=True)
        gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    except:
        print("Failed to get gpu_id (setting gpu_id to %d)" % opt.gpu_id)
        gpu_id = str(opt.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    opt.logger.warn('GPU ID: %s | available memory: %dM' \
                    % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))

    import torch
    from torch.autograd import Variable
    from loader import textDataLoader
    import models.setup as ms
    from models.evaluate import evaluate_model
    import utils
    import utils.logging as lg
    from tensorboardX import SummaryWriter

    # reproducibility:
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    tb_writer = SummaryWriter(opt.eventname)
    # tb_writer = tf.summary.FileWriter(opt.eventname)
    opt.logger.warn('Running in dev branch')
    opt.logger.info('Reading data ...')
    src_loader = textDataLoader({'h5_file': opt.input_data_src+'.h5',
                                 'infos_file': opt.input_data_src+'.infos',
                                 "max_seq_length": opt.max_src_length,
                                 'batch_size': opt.batch_size},
                                logger=opt.logger)

    trg_loader = textDataLoader({'h5_file': opt.input_data_trg+'.h5',
                                 'infos_file': opt.input_data_trg+'.infos',
                                 "max_seq_length": opt.max_trg_length,
                                 'batch_size': opt.batch_size},
                                logger=opt.logger)

    src_vocab_size = src_loader.get_vocab_size()
    trg_vocab_size = trg_loader.get_vocab_size()
    opt.vocab_size = trg_vocab_size
    # Recover saved epoch || idempotent jobs
    iteration, epoch, opt, infos, history = ms.recover_infos(opt)
    src_loader.iterators = infos.get('src_iterators', src_loader.iterators)
    trg_loader.iterators = infos.get('trg_iterators', trg_loader.iterators)
    iteration -= 1  # start with an evaluation
    opt.logger.info('Starting from Epoch %d, iteration %d' % (epoch, iteration))
    # Recover data iterator and best perf
    src_loader.iterators = infos.get('src_iterators', src_loader.iterators)
    trg_loader.iterators = infos.get('trg_iterators', trg_loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    del infos
    model = ms.select_model(opt, src_vocab_size, trg_vocab_size)
    opt.logger.warn('Loading pretrained weights...  (batch size: %d)' % opt.batch_size)
    model.load()
    opt.logger.warn('Transferring to cuda...')
    model.cuda()
    model.define_loss(trg_loader)
    val_losses = []
    update_lr_flag = True
    optimizer = ms.set_optimizer(opt, epoch, model)
    gc.collect()
    while True:
        if update_lr_flag:
            # Assign the learning rate
            opt = utils.manage_lr(epoch, opt, val_losses)
            utils.scale_lr(optimizer, opt.scale_lr)  # set the decayed rate
            lg.log_optimizer(opt, optimizer)
            # Assign the scheduled sampling prob
            if opt.scheduled_sampling_strategy == "step":
                if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                    opt.logger.warn('ss_prob= %.2e' % model.ss_prob)
            if opt.loss_version in ['word', 'seq'] and opt.alpha_strategy == "step":
                # Update ncrit's alpha:
                opt.logger.warn('Updating the loss scaling param alpha')
                frac = epoch // opt.alpha_increase_every
                new_alpha = min(opt.alpha_increase_factor * frac, 1)
                model.crit.alpha = new_alpha
                opt.logger.warn('New alpha %.3e' % new_alpha)
            update_lr_flag = False

        if opt.scheduled_sampling_strategy == "sigmoid":
            if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                opt.logger.warn("setting up the ss_prob")
                opt.ss_prob = 1 - opt.scheduled_sampling_speed / (opt.scheduled_sampling_speed +
                                                                  exp(iteration / opt.scheduled_sampling_speed))
                model.ss_prob = opt.ss_prob
                opt.logger.warn("ss_prob =  %.3e" % model.ss_prob)
        if opt.loss_version in ['word', 'seq'] and opt.alpha_strategy == "sigmoid":
            # Update crit's alpha:
            opt.logger.warn('Updating the loss scaling param alpha')
            new_alpha = 1 - opt.alpha_speed / (opt.alpha_speed + exp(iteration / opt.alpha_speed))
            new_alpha = min(new_alpha, 1)
            model.crit.alpha = new_alpha
            opt.logger.warn('New alpha %.3e' % new_alpha)

        torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        data_src, order = src_loader.get_src_batch('train')
        tmp = [data_src['labels']]
        input_lines_src, = [Variable(torch.from_numpy(_),
                                    requires_grad=False).cuda()
                           for _ in tmp]
        src_lengths = data_src['lengths']

        data_trg = trg_loader.get_trg_batch('train', order)
        tmp = [data_trg['labels'], data_trg['out_labels'], data_trg['mask']]
        input_lines_trg, output_lines_trg, mask = [Variable(torch.from_numpy(_),
                                                            requires_grad=False).cuda()
                                                   for _ in tmp]
        trg_lengths = data_trg['lengths']
        optimizer.zero_grad()
        ml_loss, loss, stats = model.step(input_lines_src, src_lengths, input_lines_trg, trg_lengths, output_lines_trg, mask)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = []
        grad_norm.append(utils.clip_gradient(optimizer, opt.grad_clip))
        optimizer.step()
        train_loss = loss.data[0]
        train_ml_loss = ml_loss.data[0]
        if np.isnan(train_loss):
            sys.exit('Loss is nan')
        torch.cuda.synchronize()
        end = time.time()
        losses = {'train_loss': train_loss,
                  'train_ml_loss': train_ml_loss}
        lg.stderr_epoch(epoch, iteration, opt, losses, grad_norm, end-start)
        # Update the iteration and epoch
        iteration += 1
        if data_src['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            lg.log_epoch(tb_writer, iteration, opt,
                         losses, stats, grad_norm,
                         model.ss_prob)
            history['loss'][iteration] = losses['train_loss']
            history['lr'][iteration] = opt.current_lr
            history['ss_prob'][iteration] = model.ss_prob
            history['scores_stats'][iteration] = stats

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            opt.logger.warn('Evaluating the model')
            model.eval()
            eval_kwargs = {'split': 'val', 'batch_size': opt.valid_batch_size,
                           "beam_size": 1}  # otherwise slow
            _, val_ml_loss, val_loss, bleu_moses = evaluate_model(model, src_loader, trg_loader,
                                                                  opt.logger, eval_kwargs)
            opt.logger.info('Iteration : %d : BLEU: %.5f ' % (iteration, bleu_moses))
            # Write validation result into summary
            lg.add_summary_value(tb_writer, 'val_loss', val_loss, iteration)
            lg.add_summary_value(tb_writer, 'val_ml_loss', val_ml_loss, iteration)
            lg.add_summary_value(tb_writer, 'Bleu_moses', bleu_moses, iteration)
            tb_writer.file_writer.flush()
            history['val_perf'][iteration] = {'bleu': bleu_moses}
            val_losses.insert(0, val_loss)
            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = bleu_moses
            else:
                current_score = - val_loss
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            lg.save_model(model, optimizer, opt,
                          iteration, epoch, src_loader, trg_loader,
                          best_val_score,
                          history, best_flag)
            model.train()
        gc.collect()
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)

