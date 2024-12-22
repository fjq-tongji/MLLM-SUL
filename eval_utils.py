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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

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

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()
    #print(valids)   ########
    #print(len(valids))

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
    
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['word_feats'], data['attr_feats'], data['seg_feats'], data['boxes_feats'],
                   data['labels'], data['masks'], data['att_masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels[..., :-1], att_masks), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['word_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['attr_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['seg_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['boxes_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]


        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks = tmp

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # forward the model to also get generated samples for each image
        with torch.no_grad():
            final = model(fc_feats, att_feats, word_feats, attr_feats, seg_feats,
                        boxes_feats, att_masks, opt=eval_kwargs, mode='sample')
            seq = final[0].data   ###(b,L)
            #print('%%%%%%%%%%%%%%%%')
            #print(seq.shape)
            attn_c = final[-1].data   ### keshihua_decoder_cross
            attn_t = final[-2].data   ### keshihua_decoder_txt
            attn_e = final[-3].data   ### keshihua_encoder

            # print(attn[0].shape)
            # print(attn[1].shape)
            # print(attn[2].shape)
            # print(attn[3].shape)
            #visua = final[-1]       ##可视化列表


        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                #print('()()()()')
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                for _ in model.done_beams[i]:
                    seq__ = _['seq']
                    #print('--------------------')
                    #print(seq__.shape)
                    #print(seq__)
                    #print('--------------------')
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        #print('*/*/*/*/*/')
        #print(seq.shape)
        #print(sents)
        #print(type(sents)) 


        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)
            #'/home/fjq/husthuaan/log/%s.npy' % entry['image_id']
            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_encoder/baseline/' + str(entry['image_id'])), attn_e[0][0].cpu().numpy())
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_decoder_txt/baseline/' + str(entry['image_id'])), attn_t[0][0].cpu().numpy())
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_decoder_cross/baseline/' + str(entry['image_id'])), attn_c[0][0].cpu().numpy())

                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_encoder/attn_encoder_1/' + str(entry['image_id']) + '-encoder-1'), attn[0][0][0].cpu().numpy())
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_encoder/attn_encoder_2/' + str(entry['image_id']) + '-encoder-2'), attn[1][0][0].cpu().numpy())
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_encoder/attn_encoder_3/' + str(entry['image_id']) + '-encoder-3'), attn[2][0][0].cpu().numpy())
                #np.save(os.path.join('/home/fjq/husthuaan/log/attn_encoder/attn_encoder_4/' + str(entry['image_id']) + '-encoder-4'), attn[3][0][0].cpu().numpy())


        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
