import json
import pickle
import h5py
import numpy as np


class textDataLoader(object):
    """
    Text data iterator class
    """
    def __init__(self, params, logger):
        self.logger = logger
        infos = pickle.load(open(params['infos_file'], 'rb'))
        self.ix_to_word = infos['itow']
        # self.ix_to_word[0] = '<PAD>'  # already in
        self.vocab_size = len(self.ix_to_word)
        self.logger.warn('vocab size is %d ' % self.vocab_size)
        # open the hdf5 file
        self.logger.warn('DataLoader loading h5 file: %s' % params['h5_file'])
        self.h5_file = h5py.File(params['h5_file'])
        self.logger.warn('Training set length: %d' % self.h5_file['labels_train'].shape[0])
        self.logger.warn('Validation set length: %d' % self.h5_file['labels_val'].shape[0])
        self.logger.warn('Test set length: %d' % self.h5_file['labels_test'].shape[0])
        # load in the sequence data
        self.batch_size = params['batch_size']
        seq_size = self.h5_file['labels_test'].shape
        self.seq_length = seq_size[1]
        self.logger.warn('max sequence length in data is %d' % self.seq_length)
        # separate out indexes for each of the provided splits
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_src_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        label_batch = np.zeros([batch_size, self.seq_length], dtype ='int')
        ref = 'labels_%s' % split
        max_index = len(self.h5_file[ref])
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            label_batch[i] = self.h5_file[ref][ri, :self.seq_length]
        data = {}
        data['labels'] = label_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        return data

    def get_trg_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        in_label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
        out_label_batch = np.zeros([batch_size, self.seq_length], dtype='int')
        mask_batch = np.zeros([batch_size, self.seq_length], dtype='float32')

        ref = 'labels_%s' % split
        oref = 'out_labels_%s' % split
        max_index = len(self.h5_file[ref])
        wrapped = False
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            in_label_batch[i] = self.h5_file[ref][ri, :self.seq_length]
            out_label_batch[i] = self.h5_file[oref][ri, :self.seq_length]
            mask_batch[i] = self.h5_file['mask_%s' % split][ri, :self.seq_length]

        data = {}
        data['labels'] = in_label_batch
        data['out_labels'] = out_label_batch
        data['mask'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0


