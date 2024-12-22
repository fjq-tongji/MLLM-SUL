from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
#from .OldModel import ShowAttendTellModel, AllImgModel
from .ShowAttendTellModel import ShowAttendTellModel
from .AttModel import *
from .TransformerModel import TransformerModel
from .m2_trans import m2TransformerModel
#from .TransformerModel3_d1 import TransformerModel3_d1
from .AoAModel import AoAModel
#from .TransformerModel_1 import TransformerModel_1
#from .TransformerModel_2 import TransformerModel_2
#from .TransformerModel_3 import TransformerModel_3
from .TransformerModel_llama3_2_3B import TransformerModel_llama3_2_3B
from .TransformerModel_llama3_1_8B import TransformerModel_llama3_1_8B
from .TransformerModel_llama_guard_3_8B import TransformerModel_llama_guard_3_8B

#from .TransformerModel_blip2 import TransformerModel_blip2
#from .TransformerModel_instructblip2 import TransformerModel_instructblip2
#from .TransformerModel_instructblip2_2 import TransformerModel_instructblip2_2
#from .TransformerModel_blip2_t5 import TransformerModel_blip2_t5
#from .TransformerModel_llava import TransformerModel_llava
from .GSRModel import GSRModel
from .ImageTransModel import ImageTransModel
from .NSAN import NSAN
from .M2Transformer import M2TransformerModel
from .TransformerModel_LW import LWTransformerModel

def setup(opt):
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'language_model':
        model = LMModel(opt)
    elif opt.caption_model == 'newfc':
        model = NewFCModel(opt)
    elif opt.caption_model == 'GSRModel':
        model = GSRModel(opt)
    elif opt.caption_model == 'ImageTransModel':
        model = ImageTransModel(opt)
    elif opt.caption_model == 'NSAN':
        model = NSAN(opt)
    elif opt.caption_model == 'm2_trans':
        model = M2TransformerModel(opt)

    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # Transformer
    #elif opt.caption_model == 'transformer':
     #   model = TransformerModel(opt)
    #elif opt.caption_model == 'transformer_1':
     #   model = TransformerModel_1(opt)
    # elif opt.caption_model == 'transformer_2':
    #     model = TransformerModel_2(opt)
    elif opt.caption_model == 'transformer_lw':
        model = LWTransformerModel(opt)
    # elif opt.caption_model == 'transformer_3':
    #     model = TransformerModel_3(opt)
    elif opt.caption_model == 'transformer_llama3_2_3B':
        model = TransformerModel_llama3_2_3B(opt)
    elif opt.caption_model == 'transformer_llama3_1_8B':
        model = TransformerModel_llama3_1_8B(opt)
    elif opt.caption_model == 'transformer_llama_guard_3_8B':
                model = TransformerModel_llama_guard_3_8B(opt)
    # elif opt.caption_model == 'transformer_blip2':
    #     model = TransformerModel_blip2(opt)
    # elif opt.caption_model == 'transformer_blip2_t5':
    #     model = TransformerModel_blip2_t5(opt)
    # elif opt.caption_model == 'transformer_llava':
    #     model = TransformerModel_llava(opt)
    # elif opt.caption_model == 'transformer_instructblip2':
    #     model = TransformerModel_instructblip2(opt)
    # elif opt.caption_model == 'transformer_instructblip2_2':
    #     model = TransformerModel_instructblip2_2(opt)




    elif opt.caption_model == 'm2transformer':
        model = m2TransformerModel(opt)
    #elif opt.caption_model == 'image_transformer':
     #   model = TransformerModel3_d1(opt)

    # AoANet
    elif opt.caption_model == 'aoa':
        model = AoAModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model
