from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import lmdb
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing
import six

import opts

#opt = opts.parse_opt()

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """
    def __init__(self, db_path, ext, feature_type):
        self.db_path = db_path
        self.ext = ext
        self.feature_type = feature_type
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x)
        else:
            if 'cocotest' in self.db_path:
                self.loader = lambda x: np.load(x)['z']
            else:
                if self.feature_type == 'att_feats':
                    self.loader = lambda x: np.load(x)['feat']
                if self.feature_type == 'grid_feats':
                    self.loader = lambda x: np.load(x)['features']   ###['g_feature'] 是global特征
                if self.feature_type == 'global_feats':
                    self.loader = lambda x: np.load(x)['features']   ###['g_feature'] 是global特征
                if self.feature_type == 'semantic_feats':
                    self.loader = lambda x: np.load(x)


        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        else:
            self.db_type = 'dir'
    
    def get(self, key):

        if self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key)
            f_input = six.BytesIO(byteflow)
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        else:
            f_input = os.path.join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)

        return feat


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    # def get_vocab_size(self):
    #     return self.vocab_size
    #
    # def get_vocab(self):
    #     return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        #self.dataset = Dataset(opt)

        # feature related options
        self.use_fc = getattr(opt, 'use_fc', False)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        self.norm_grid_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_global_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_semantic_feat = getattr(opt, 'norm_att_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        # if 'ix_to_word' in self.info:
        #     self.ix_to_word = self.info['ix_to_word']
        #     self.vocab_size = len(self.ix_to_word)
        #     print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape    #### 16, 150
            self.tokens = self.h5_label_file['labels'][:]    #### 问题+答案
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.tokens_start_ix = self.h5_label_file['label_start_ix'][:]
            self.tokens_end_ix = self.h5_label_file['label_end_ix'][:]

            #-------------------------------------------------------------------------# 把回归的坐标框数字加进来
            reg_size = self.h5_label_file['bbox'].shape         #### (1,4)
            self.regs = self.h5_label_file['bbox'][:]
            self.reg_length = reg_size[1]
            #-------------------------------------------------------------------------#

        else:
            self.seq_length = 1

        #self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', 'fc_feats')
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', 'att_feats')
        #self.word_feats_loader = HybridLoader(self.opt.input_word_dir, '.npy', 'word_feats')
        #self.attr_feats_loader = HybridLoader(self.opt.input_attr_dir, '.npy', 'attr_feats')
        self.grid_feats_loader = HybridLoader(self.opt.input_grid_dir, '.npz', 'grid_feats')
        self.global_feats_loader = HybridLoader(self.opt.input_global_dir, '.npz', 'global_feats')
        self.semantic_feats_loader = HybridLoader(self.opt.input_semantic_dir, '.npy', 'semantic_feats')

        self.num_images = len(self.info['images'])
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.tokens_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.tokens_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            tokens = np.zeros([seq_per_img, self.seq_length], dtype = 'int')    ###(5,16) 0
            regs = np.zeros([seq_per_img, 4], dtype='float')  ###(5,4)
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.tokens[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            tokens = self.tokens[ixl: ixl + seq_per_img, :self.seq_length]
            regs = self.regs[ixl: ixl + seq_per_img, :4]

        return tokens, regs        ###(1,16)   (1,4)


    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        #cls_token_batch = []
        tokens_batch = [] #np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        regs_batch = []
        #word_feats_batch = []   #np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        #attr_feats_batch = []
        grid_feats_batch = []
        global_feats_batch = []
        semantic_feats_batch = []

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_grid_feats, tmp_global_feats, tmp_semantic_feats, tmp_seq, tmp_bbox, ix, tmp_wrapped = self._prefetch_process[split].get()                 ##************************************
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)     #### att_batch B*2048
            #cls_token_batch.append(tmp_cls_token)
            #word_feats_batch.append(tmp_word_feats)                                  ##************************************
            #attr_feats_batch.append(tmp_attr_feats)
            grid_feats_batch.append(tmp_grid_feats)
            global_feats_batch.append(tmp_global_feats)
            semantic_feats_batch.append(tmp_semantic_feats)

            tmp_tokens = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
               tmp_tokens[:, 0:self.seq_length] = tmp_seq    ####补了一下<bos>和<eos>
            tokens_batch.append(tmp_tokens)      ###tmp_tokens


            #--------------------------------------------------------------------------------------------------------#
            tmp_regs = np.zeros([seq_per_img, 4], dtype = 'float32')     ######(1,4)
            tmp_regs[:, 0:4] = tmp_bbox
            regs_batch.append(tmp_regs)
            #--------------------------------------------------------------------------------------------------------#


            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                gts.append(self.tokens[self.tokens_start_ix[ix] - 1: self.tokens_end_ix[ix]])
            else:
                gts.append([])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['gt_bbox'] = self.info['images'][ix]['bbox']
            #info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, tokens_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(tokens_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        if self.use_box:
            fc_batch, att_batch, grid_feats_batch, global_feats_batch, semantic_feats_batch, tokens_batch, regs_batch, gts, infos = \
                zip(*sorted(zip(fc_batch, att_batch, grid_feats_batch, global_feats_batch, semantic_feats_batch, tokens_batch, regs_batch, gts, infos), key=lambda x: 0, reverse=True))
        else:
            fc_batch, att_batch, grid_feats_batch, global_feats_batch, semantic_feats_batch, tokens_batch, regs_batch, gts, infos = \
                zip(*sorted(zip(fc_batch, att_batch, grid_feats_batch, global_feats_batch, semantic_feats_batch, tokens_batch, regs_batch, gts, infos), key=lambda x: 0, reverse=True))


        data = {}
        data['fc_feats'] = np.stack(sum([[_]*seq_per_img for _ in fc_batch], []))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        #max_word_len = max(_.shape[0] for _ in word_feats_batch)
        #max_attr_len = max(_.shape[0] for _ in attr_feats_batch)
        max_grid_len = max([_.shape[0] for _ in grid_feats_batch])
        max_global_len = max([_.shape[0] for _ in global_feats_batch])
        max_semantic_len = max([_.shape[0] for _ in semantic_feats_batch])

        data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, max_att_len, att_batch[0].shape[1]], dtype = 'float32')    ## (5,B,2048)
        #data['cls_token_feats'] = np.zeros([len(cls_token_batch) * seq_per_img, 1, cls_token_batch[0].shape[1]],
         #                            dtype='float32')      ## (5,1,768)
        #data['word_feats'] = np.zeros([len(att_batch)*seq_per_img, max_word_len], dtype = 'int32')      ## (5,B)
        #data['attr_feats'] = np.zeros([len(att_batch)*seq_per_img, max_attr_len], dtype = 'int32')
        data['grid_feats'] = np.zeros([len(grid_feats_batch)*seq_per_img, max_grid_len, grid_feats_batch[0].shape[1]], dtype = 'float32')
        data['global_feats'] = np.zeros([len(global_feats_batch) * seq_per_img, max_global_len, global_feats_batch[0].shape[1]],
                                      dtype='float32')
        data['semantic_feats'] = np.zeros(
            [len(semantic_feats_batch) * seq_per_img, max_semantic_len], dtype='int32')


        for i in range(len(att_batch)):
            data['att_feats'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
            #data['cls_token_feats'][i * seq_per_img:(i + 1) * seq_per_img, :cls_token_batch[i].shape[0]] = cls_token_batch[i]
            #print('---------')
            #print(att_batch[i].shape[0])
            # if word_feats_batch[i].shape[0] > max_word_len:
            #     data['word_feats'][i * seq_per_img:(i + 1) * seq_per_img, :max_word_len] = word_feats_batch[i][:max_word_len]
            # else:
            #     data['word_feats'][i * seq_per_img:(i + 1) * seq_per_img, :word_feats_batch[i].shape[0]] = word_feats_batch[i]
            #
            # if attr_feats_batch[i].shape[0] > max_attr_len:
            #     data['attr_feats'][i * seq_per_img:(i + 1) * seq_per_img, :max_attr_len] = attr_feats_batch[i][:max_attr_len]
            # else:
            #     data['attr_feats'][i * seq_per_img:(i + 1) * seq_per_img, :attr_feats_batch[i].shape[0]] = attr_feats_batch[i]

            if grid_feats_batch[i].shape[0] > max_grid_len:
                data['grid_feats'][i * seq_per_img:(i + 1) * seq_per_img, :max_grid_len] = grid_feats_batch[i][:max_grid_len]
            else:
                data['grid_feats'][i * seq_per_img:(i + 1) * seq_per_img, :grid_feats_batch[i].shape[0]] = grid_feats_batch[i]

            if global_feats_batch[i].shape[0] > max_global_len:
                data['global_feats'][i * seq_per_img:(i + 1) * seq_per_img, :max_global_len] = global_feats_batch[i][:max_global_len]
            else:
                data['global_feats'][i * seq_per_img:(i + 1) * seq_per_img, :global_feats_batch[i].shape[0]] = global_feats_batch[i]

            if semantic_feats_batch[i].shape[0] > max_semantic_len:
                data['semantic_feats'][i * seq_per_img:(i + 1) * seq_per_img, :max_semantic_len] = semantic_feats_batch[i][:max_semantic_len]
            else:
                data['semantic_feats'][i * seq_per_img:(i + 1) * seq_per_img, :semantic_feats_batch[i].shape[0]] = semantic_feats_batch[i]


        if self.use_box:
            data['boxes_feats'] = np.zeros([len(boxes_batch) * seq_per_img, max_att_len, boxes_batch[0].shape[1]], dtype='float32')  ## (5,B,4)


        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['tokens'] = np.vstack(tokens_batch)
        data['regs'] = np.vstack(regs_batch)

        #print(data['tokens'].shape)   #### (10, 18)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['tokens'])))
        mask_batch = np.zeros([data['tokens'].shape[0], self.seq_length + 2], dtype = 'float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        #print(data['masks'].shape)


        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        #data = {k:torch.tensor(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data_2 = {}
        for k, v in data.items():
            if type(v) is np.ndarray and k != 'regs':
                data_2[k] = torch.tensor(v)
            elif type(v) is np.ndarray and k == 'regs':
                torch.set_printoptions(precision=4)
                data_2[k] = torch.tensor(v, dtype=torch.float32)
            else:
                data_2[k] = v

        #print('-------------------------------')
        #print(data['word_feats'].shape)  (5,B)
        #print(data['att_feats'].shape)   (5,B,2048)
        return data_2

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):   ###迭代器
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))    ## 下载npz文件
            grid_feats = self.grid_feats_loader.get(str(self.info['images'][ix]['id']))  ## 下载npz文件
            global_feats = self.global_feats_loader.get(str(self.info['images'][ix]['id']))  ## 下载npz文件
            semantic_feats = self.semantic_feats_loader.get(str(self.info['images'][ix]['id']))  ## 下载npy文件
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.norm_grid_feat:
                grid_feats = grid_feats / np.linalg.norm(grid_feats, 2, 1, keepdims=True)
            if self.norm_global_feat:
                global_feats = global_feats / np.linalg.norm(global_feats, 2, 1, keepdims=True)
            if self.norm_semantic_feat:
                semantic_feats = semantic_feats / np.linalg.norm(semantic_feats, 2, 1, keepdims=True)


            #word_feats = self.word_feats_loader.get(str(self.info['images'][ix]['id']))   ## 下载npy文件   ##************************************
            #attr_feats = self.attr_feats_loader.get(str(self.info['images'][ix]['id']))
            #cls_token_feats = self.cls_token_loader.get(str(self.info['images'][ix]['id']))

            if self.use_box:
                boxes_feat = self.box_loader.get(str(self.info['images'][ix]['id']))

                # devided by image width and height
                #x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                #h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                #box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                #if self.norm_box_feat:
                 #   box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                #att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                #att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((1,1,1), dtype='float32')
            #word_feats = self.word_feats_loader.get(str(self.info['images'][ix]['id']))  ## 下载npy文件   ##************************************
            #attr_feats = self.attr_feats_loader.get(str(self.info['images'][ix]['id']))
            grid_feats = self.grid_feats_loader.get(str(self.info['images'][ix]['id']))
            global_feats = self.global_feats_loader.get(str(self.info['images'][ix]['id']))
            semantic_feats = self.semantic_feats_loader.get(str(self.info['images'][ix]['id']))
            boxes_feat = self.box_loader.get(str(self.info['images'][ix]['id']))

        # if self.use_fc:
        #     fc_feat = np.zeros((1), dtype='float32')    ###self.fc_loader.get(str(self.info['images'][ix]['id']))
        # else:

        fc_feat = np.zeros((1), dtype='float32')

        if hasattr(self, 'h5_label_file'):
            tokens, regs = self.get_captions(ix, self.seq_per_img)      ####(1,16) int,  (1,4) float
        else:
            tokens = None
            regs = None


        if self.use_box:
            return (fc_feat,
                    att_feat, grid_feats, global_feats, semantic_feats, boxes_feat, tokens, regs,
                    ix)
        else:
            return (fc_feat,
                    att_feat, grid_feats, global_feats, semantic_feats, tokens, regs,
                    ix)

    def __len__(self):
        return len(self.info['images'])












class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]













