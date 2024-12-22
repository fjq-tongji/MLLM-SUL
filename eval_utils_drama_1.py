from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import torch
from tokenizer import Tokenizer
import cv2

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'coco-caption/f30k_captions4eval.json'
    elif 'cityscapes' in dataset:                   ###### add 
        annFile = 'data/cityscapes/cityscapes_captions4eval.json'
    elif 'camvid' in dataset:
        annFile = 'data/camvid/CamVid_captions4eval.json'
    elif 'FoggyCityscapes' in dataset:
        annFile = 'data/FoggyCityscapes/FoggyCityscapes_captions4eval.json'
    elif 'drama' in dataset:
        annFile = 'data/drama/drama_captions4eval.json'
        #annFile = 'data/drama/rolisp_dataset_compare/drama_captions4eval_ROLISP.json'      ####only response

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    
    #out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out



def eval_split(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to also get generated samples for each image
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['grid_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['global_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['semantic_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['tokens'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['regs'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]

        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens, regs, att_masks = tmp


        # forward the model to also get generated samples for each image
        with torch.no_grad():
            sents = model(fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens,
                        att_masks, opt=eval_kwargs, mode='sample')                      ###(b,120)  (b,4)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}    ####only response
            predictions.append(entry)
            print('image %s: %s' % (entry['image_id'], entry['caption']))


        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
           predictions.pop()
        print('evaluating validation preformance... %d/%d' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return predictions, lang_stats




