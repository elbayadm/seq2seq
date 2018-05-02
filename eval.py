# -*- coding: utf-8 -*-
"""
Main evaluation script
"""

import os
import os.path as osp
import time
import pickle
import json
import random
import numpy as np
from options import parse_eval_opt, get_gpu_memory


if __name__ == "__main__":
    opt = parse_eval_opt()
    if not opt.output:
        evaldir = '%s/evaluations/%s' % (opt.modelname, opt.split)
        if not osp.exists(evaldir):
            os.makedirs(evaldir)
        opt.output = '%s/bw%d' % (evaldir, opt.beam_size)
        # if opt.beam_size == 1:
            # sampling = "_samplemax" if opt.sample_max else "_sample_temp_%.3f" % opt.temperature
            # opt.output += sampling
    if not osp.exists(opt.output + '.json'):
        try:
            import subprocess
            # gpu_id = subproces.check_output('source gpu_setVisibleDevices.sh', shell=True)
            gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            opt.logger.warn('GPU ID: %s | available memory: %dM' \
                            % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))

        except:
            opt.logger.warn("Requested gpu_id : %s" % opt.gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
            opt.logger.warn('GPU ID: %s | available memory: %dM' \
                            % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(opt.gpu_id)))
        import torch
        from models.evaluate import evaluate_model
        from torch.autograd import Variable
        from loader import textDataLoader
        import models.setup as ms
        import utils

        print('Reading data ...')
        src_loader = textDataLoader({'h5_file': opt.input_data_src+'.h5',
                                     'infos_file': opt.input_data_src+'.infos',
                                     'max_seq_length': opt.max_src_length,
                                     'batch_size': opt.batch_size},
                                    logger=opt.logger)

        trg_loader = textDataLoader({'h5_file': opt.input_data_trg+'.h5',
                                     'infos_file': opt.input_data_trg+'.infos',
                                     'max_seq_length': opt.max_trg_length,
                                     'batch_size': opt.batch_size},
                                    logger=opt.logger)

        src_vocab_size = src_loader.get_vocab_size()
        trg_vocab_size = trg_loader.get_vocab_size()
        opt.vocab_size = trg_vocab_size

        # reproducibility:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)

        if opt.start_from_best:
            flag = '-best'
            opt.logger.warn('Starting from the best saved model')
        else:
            flag = ''
        opt.infos_start_from = osp.join(opt.modelname, 'infos%s.pkl' % flag)
        opt.start_from = osp.join(opt.modelname, 'model%s.pth' % flag)
        opt.logger.warn('Starting from %s' % opt.start_from)

        # Load infos
        with open(opt.infos_start_from, 'rb') as f:
            print('Opening %s' % opt.infos_start_from)
            infos = pickle.load(f, encoding="iso-8859-1")
            # update special tokens (especially for old models trained with a different vocab)
            # trg_vocab = {w: k for k, w in infos['trg_vocab'].items()}
            # trg_loader.eos = trg_vocab['<EOS>']
            # trg_loader.bos = trg_vocab['<BOS>']
            # trg_loader.unk = trg_vocab['<UNK>']
            # trg_loader.pad = trg_vocab['<PAD>']

            infos['opt'].logger = None
        ignore = ["batch_size", "beam_size", "start_from",
                  'infos_start_from', "split",
                  "start_from_best", "language_eval", "logger",
                  "val_images_use", 'input_data', "loss_version",
                  "clip_reward",
                  "gpu_id", "max_epochs", "modelname"]

        for k in list(vars(infos['opt']).keys()):
            if k not in ignore:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

        # Avoid setting up intricate losses for no reason:
        opt.loss_version = "ml"
        model = ms.select_model(opt, src_vocab_size, trg_vocab_size)
        model.load()
        opt.logger.warn('Transferring to cuda...')
        model.cuda()
        opt.logger.info('Setting up the loss function %s' % opt.loss_version)
        model.define_loss(trg_loader)
        opt.logger.warn('Evaluating the model')
        model.eval()
        eval_kwargs = {"beam_size": 1,
                       "verbose": 1}
        eval_kwargs.update(vars(opt))
        start = time.time()
        for k in ['start_from_best', 'beam_size', 'split']:
            print(k, eval_kwargs[k])
        # print('Eval args:', eval_kwargs)
        preds, ml_loss, loss, bleu_moses = evaluate_model(model, src_loader, trg_loader,
                                                          opt.logger, eval_kwargs)
        print("Finished evaluation in ", (time.time() - start))
        # print('ML loss:', ml_loss)
        # print('Training loss:', loss)
        perf = {}
        perf['Bleu'] = bleu_moses
        perf['loss'] = loss
        print('Results:', perf)
        perf['ml_loss'] = ml_loss
        perf['params'] = eval_kwargs
        perf['params']['logger'] = None

        pickle.dump(perf, open(opt.output + ".res", 'wb'))
        if opt.dump_json:
            json.dump(preds, open(opt.output + '.json', 'w', encoding='utf8'), ensure_ascii=False)
            # json.dump(preds, open(opt.output + '.json', 'w'))

    else:
        # ----------- Score pre-existing predictions.
        from models.evaluate import score_trads
        from loader import textDataLoader
        opt.logger.warn('Evaluating already generated caps')
        # Possibly you've already evaluated the lossess:
        try:
            perf = pickle.load(open(opt.output + '.res', 'rb'))
        except:
            opt.logger.error('No pre-existing evaluation')
            perf = {}
        trg_loader = textDataLoader({'h5_file': opt.input_data_trg+'.h5',
                                     'infos_file': opt.input_data_trg+'.infos',
                                     'max_seq_length': opt.max_trg_length,
                                     'batch_size': opt.batch_size},
                                    logger=opt.logger)
        eval_kwargs = {"split": opt.split,
                       "beam_size": opt.beam_size,
                       "verbose": 1,
                       "batch_size" : opt.batch_size}
        eval_kwargs.update(vars(opt))
        preds = json.load(open(opt.output + '.json', 'r'))
        scores = score_trads(preds, trg_loader, eval_kwargs)
        perf.update(scores)
        print('Performances:', perf)
        pickle.dump(perf, open(opt.output + ".res", 'wb'))


