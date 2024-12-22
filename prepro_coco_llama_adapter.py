import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from imageio import imread
#from scipy import imresize
from PIL import Image
from tokenizer import Tokenizer
import torch


def assign_splits(imgs, params):
    num_val = params['num_val']
    num_test = params['num_test']

    for i, img in enumerate(imgs):
        if i < num_val:
            img['split'] = 'val'
        elif i < num_val + num_test:
            img['split'] = 'test'
        else:
            img['split'] = 'train'

    print('assigned %d to val, %d to test.' % (num_val, num_test))


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


def encode_captions(imgs, params):

    max_length = params['max_length']
    N = len(imgs)    ###123287
    M = sum(len(img['sentences']) for img in imgs)  # total number of captions  12387*5

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1

    #-------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------------------------------------------------------#
    format_instruction = "Generate caption of this image"              ########################
    input1 = format_prompt(format_instruction, None)
    model_path = '/data/fjq/LLAMA/llama-2-7b/tokenizer.model'

    for i, img in enumerate(imgs):     ###img -> dict
        n = len(img['sentences'])   ###5
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')       ##(5,60)
        for j, s in enumerate(img['sentences']):
            label_length[caption_counter] = min(max_length, len(s['raw']))  # record the length of this sequence
            caption_counter += 1
            s_instuction_text = input1 + s['raw']
            #print(s_instuction_text)
            input2 = torch.tensor(Tokenizer(model_path).encode(s_instuction_text, bos=True, eos=True), dtype=torch.int64)
                                          ###(16)  tensor
            padding = max_length - input2.shape[0]
            if padding > 0:
                input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))   ####会有-1
            elif padding < 0:
                input2 = input2[:max_length]
            print(input2.shape)
            print(input2)
            input2_mask = input2.ge(0)
            input2[~input2_mask] = 0      ####-1都变成了0
            input2_new = input2.numpy()
            Li[j] = input2_new
            print(Li[j].shape)
            #print(Li[j])
        print(Li.shape)
    # ------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------------------------------------------------------#

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    print('$$$$$$$$$$$$$$$$$')
    print(L.shape)
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    #assert np.all(label_length > 0), 'error: some caption had no words?'

    #print('encoded captions to array of size ', `L.shape`)
    return L, label_start_ix, label_end_ix, label_length




def main(params):
    input_json = json.load(open(params['input_json'], 'r'))
    imgs = input_json['images']
    #shuffle(imgs)  # shuffle the order

    seed(123)  # make reproducible

    # # encode captions in large arrays, ready to ship to hdf5 file
    #-------------------------------------------------------------------------------------------------------------------#生成h5文件
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params)
    print('//////////\\\\\\\\\\\\\\\\\\')
    print(L.shape)    #####(5,16)
    print(L)
    #
    # # create output h5 file
    # f = h5py.File(params['output_h5'], "w")
    # f.create_dataset("labels", dtype='uint32', data=L)
    # f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    # f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    # f.create_dataset("label_length", dtype='uint32', data=label_length)
    # f.close()
    # print('wrote ', params['output_h5'])
    #-------------------------------------------------------------------------------------------------------------------#


    #-------------------------------------------------------------------------------------------------------------------#生成json文件
    out = {}
    #out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filepath' in img: jimg['file_path'] = img['filepath']  # copy it over, might need
        if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])
    #-------------------------------------------------------------------------------------------------------------------#




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default=r'data/coco/dataset_coco.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--num_val', default=5000,  type=int,
                        help='number of images to assign to validation data (for CV etc)')
    parser.add_argument('--output_json', default=r'data/coco/coco_llama_adapter.json',
                        help='output json file')
    parser.add_argument('--output_h5', default=r'data/coco/coco_llama_adapter.h5',
                        help='output h5 file')

    # options
    parser.add_argument('--max_length', default=60, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--num_test', default=5000, type=int,
                        help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)










