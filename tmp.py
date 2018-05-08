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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        opt.logger.warn('GPU ID: %s | available memory: %dM' \
                        % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))

    except:
        opt.logger.warn("Requested gpu_id : %s" % opt.gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        opt.logger.warn('GPU ID: %s | available memory: %dM' \
                        % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(opt.gpu_id)))


    from loader import textDataLoader
    from utils import decode_sequence

    # reproducibility:
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

    goon = True
    bound = 0
    while goon:
        # Load data from train split (0)
        data_src, order = src_loader.get_src_batch('test')
        input_lines_src = data_src['labels']
        data_trg = trg_loader.get_trg_batch('test', order)
        output_lines_trg = data_trg['out_labels']
        sent_source = decode_sequence(src_loader.get_vocab(),
                                      input_lines_src,
                                      eos=src_loader.eos, bos=src_loader.bos)
        sent_gold = decode_sequence(trg_loader.get_vocab(),
                                    output_lines_trg,
                                    eos=trg_loader.eos,
                                    bos=trg_loader.bos)

        for i, (src, trg) in enumerate(zip(sent_source, sent_gold)):
            if bound + i in [134, 1924, 2092]:
                print(bound + i, '>>>')
                print('Source:', src)
                print('Target:', trg)
        bound = data_src['bounds']['it_pos_now']
        goon = bound < 2100


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)

