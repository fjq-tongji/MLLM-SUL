from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle
import traceback

import opts
import models
from dataloader_drama_3 import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
#from misc.loss_wrapper import LossWrapper
import fvcore
from fvcore.nn import parameter_count_table
import copy


# os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def compute_param(model):
    total = sum([param.nelement() for param in model.model.encoder.layers.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def Giou_(box1, box2):    ##(b,4)
    _bs = box1.shape[0]
    box11 = torch.zeros(_bs, 4).cuda()
    box22 = torch.zeros(_bs, 4).cuda()
    box11[:, 0] = box1[:, 0]
    box11[:, 1] = box1[:, 1] - box1[:, 3]
    box11[:, 2] = box1[:, 0] + box1[:, 2]
    box11[:, 3] = box1[:, 1]
    box22[:, 0] = box2[:, 0]
    box22[:, 1] = box2[:, 1] - box2[:, 3]
    box22[:, 2] = box2[:, 0] + box2[:, 2]
    box22[:, 3] = box2[:, 1]
    print(box11[0, :])
    print(box22[0, :])

    # 计算两个框的面积
    area_box11 = (box11[:, 2] - box11[:, 0]) * (box11[:, 3] - box11[:, 1])
    area_box22 = (box22[:, 2] - box22[:, 0]) * (box22[:, 3] - box22[:, 1])

    # 计算交集框的坐标
    max_xy = torch.min(box11[:, 2:], box22[:, 2:])
    min_xy = torch.max(box11[:, :2], box22[:, :2])

    # 计算交集框的面积
    inter = torch.clamp((max_xy - min_xy), min=0)  # 确保tensor的下限是0
    inter = inter[:, 0] * inter[:, 1]

    # 计算并集框的面积
    union_area = area_box11 + area_box22 - inter

    # 最小包围框C
    enclose_left_up = torch.min(box11[:, :2], box22[:, :2])
    enclose_right_down = torch.max(box11[:, 2:], box22[:, 2:])
    enclose = torch.clamp((enclose_right_down - enclose_left_up), min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1]

    # 计算 GIoU 损失
    iou_ = inter / union_area
    gious = iou_ - 1.0 * (enclose_area - union_area) / enclose_area
    giou_loss = ((1 - gious).sum())/_bs

    return giou_loss


def Iou_(box1, box2):    ##(b,4)
    _bs = box1.shape[0]
    box11 = torch.zeros(_bs, 4).cuda()
    box22 = torch.zeros(_bs, 4).cuda()
    box11[:, 0] = box1[:, 0]
    box11[:, 1] = box1[:, 1] - box1[:, 3]
    box11[:, 2] = box1[:, 0] + box1[:, 2]
    box11[:, 3] = box1[:, 1]
    box22[:, 0] = box2[:, 0]
    box22[:, 1] = box2[:, 1] - box2[:, 3]
    box22[:, 2] = box2[:, 0] + box2[:, 2]
    box22[:, 3] = box2[:, 1]
    print(box11[0, :])
    print(box22[0, :])

    # 计算两个框的面积
    area_box11 = (box11[:, 2] - box11[:, 0]) * (box11[:, 3] - box11[:, 1])
    area_box22 = (box22[:, 2] - box22[:, 0]) * (box22[:, 3] - box22[:, 1])

    # 计算交集框的坐标
    max_xy = torch.min(box11[:, 2:], box22[:, 2:])
    min_xy = torch.max(box11[:, :2], box22[:, :2])

    # 计算交集框的面积
    inter = torch.clamp((max_xy - min_xy), min=0)  # 确保tensor的下限是0
    inter = inter[:, 0] * inter[:, 1]

    # 计算并集框的面积
    union_area = area_box11 + area_box22 - inter + 1e-7
    iou_ = inter / union_area
    iou_ = iou_.sum() / _bs

    return iou_



def train(opt):
    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                utils.pickle_dump(histories, f)



    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#----------------------------------------------------------------------------------------------------------------------#

    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    #if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    acc_steps = getattr(opt, 'acc_steps', 1)
        
    loader = DataLoader(opt)
    #opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        #infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    #opt.vocab = loader.get_vocab()
    model = models.setup(opt)   #.cuda()
    print(parameter_count_table(model))    ########## 计算参数表格
    print(model)      ######## 输出模型各层的名称

#------------------------------------------------------------------------------------------------------------------------#
    if torch.cuda.device_count() > 1:
        dp_lw_model = torch.nn.DataParallel(model).cuda()
    else:
        dp_lw_model = model.cuda()
#---------------------------------------------------------------------------------------------# 把LLAMA-7B的参数喂进到模型中
    checkpoint_llama = torch.load(opt.pretrained_llama, map_location='cpu')  #***************************
    dp_lw_model.load_state_dict(checkpoint_llama, strict=False)                 ###model
    print('************')
    print('Pre-training LLAMA weights OK!')
    print('************')
# -----------------------------------------------------------------------------------------------------------------------#

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    #optimizer = utils.build_optimizer(model.parameters(), opt)
    model_parameters_filter = filter(lambda p: p.requires_grad, model.parameters())   ###289个参数不用训练
    optimizer = utils.build_optimizer(model_parameters_filter, opt)

    #---------------------------------------------------------------------------------#
    #param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    #optimizer = torch.optim.AdamW(param_groups, lr=opt.learning_rate, betas=(0.9, 0.95))
    # ---------------------------------------------------------------------------------#
    param_ = add_weight_decay(dp_lw_model)


    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))


    giou_lst = []
    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        start = time.time()
        if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
            opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
            utils.set_lr(optimizer, opt.current_lr)
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        if (iteration % acc_steps == 0):
            optimizer.zero_grad()

        torch.cuda.synchronize()
        start = time.time()
        tmp = [data['fc_feats'], data['att_feats'], data['grid_feats'], data['global_feats'], data['semantic_feats'], data['tokens'], data['regs'], data['masks'], data['att_masks']]

        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens, regs, masks, att_masks = tmp

        model_out = dp_lw_model(fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens, att_masks)
        model_out_cap = model_out[0]  ###(b,L,128256)
        model_out_reg = model_out[1]  ###(b,4)

        #---------------------------------------------------------------------------------------------------------------#制作标签
        #---------------------------------------------------------------------------------------------------------------#
        tokens_target = copy.deepcopy(tokens)

        length = 54   ###DRAMA-sris数据集=64    coco-42   DRAMA数据集=44   llama3问题部分长度为54
        tokens_target[:, :length] = -1
        target_mask = tokens_target.ge(0)
        tokens_target[~target_mask] = 0         ###tokens_target(5,60)
        # --------------------------------------------------------------------------------------------------------------#
        # --------------------------------------------------------------------------------------------------------------#

        creterion = torch.nn.CrossEntropyLoss(ignore_index=0)    #### 0被忽略
        sm_l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        lambda_ = 10.0
        out_ = model_out_cap[:, :-1]
        labels_ = tokens_target[:, 1:]
        loss_cap = creterion(out_.reshape(-1, 128256), labels_.flatten())        ####描述损失
        regs = regs

        loss_x1 = sm_l1_loss(model_out_reg[:, 0], regs[:, 0])
        loss_x2 = sm_l1_loss(model_out_reg[:, 1], regs[:, 1])
        loss_y1 = sm_l1_loss(model_out_reg[:, 2], regs[:, 2])
        loss_y2 = sm_l1_loss(model_out_reg[:, 3], regs[:, 3])
        loss_reg = loss_x1+loss_x2+loss_y1+loss_y2
        # print(Giou_(model_out_reg, regs))
        giou_loss = Giou_(model_out_reg, regs)

        loss = loss_reg + giou_loss  # lambda_ * loss_reg

        giou_lst.append(Iou_(model_out_reg, regs).item())

        loss_sp = loss / acc_steps
        loss_sp.backward()
        torch.cuda.empty_cache()

        if opt.grad_clip_value != 0:
            getattr(torch.nn.utils, 'clip_grad_%s_' %(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
        ### model.parameters()
        #=========================================================================================#
        optimizer.step()
        torch.cuda.synchronize()
        train_loss = loss.item()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.5f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.5f}, time/batch = {:.3f}" \
                .format(iteration, epoch, model_out_cap['reward'].mean(), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            iou_avg = sum(giou_lst) / len(giou_lst)
            print('---------------------trainset_iou_avg_epoch-------------------------------')
            print(iou_avg)
            giou_lst = []
            epoch += 1
            epoch_done = True

            print('---------------------testset_iou_avg_epoch-------------------------------')
            giou_test_lst = []
            thre = 0.5
            giou_test_lst_thre = []
            number_test = 0
            batch_size_test = 10
            num_images_test_total = 2544
            while True:
                data = loader.get_batch(split='test', batch_size=batch_size_test)
                number_test += batch_size_test
                tmp = [data['fc_feats'], data['att_feats'], data['grid_feats'], data['global_feats'],
                       data['semantic_feats'], data['tokens'], data['regs'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else _.cuda() for _ in tmp]
                fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens, regs, masks, att_masks = tmp
                with torch.no_grad():
                    model_out = dp_lw_model(fc_feats, att_feats, grid_feats, global_feats, semantic_feats, tokens, att_masks)
                model_out_reg_test = model_out[1]  ###(b,4)
                print(model_out_reg_test.shape)
                gts_test = regs
                iou_ = Iou_(model_out_reg_test, gts_test).item()
                giou_test_lst.append(iou_)
                if iou_ > thre:
                    giou_test_lst_thre.append(iou_)
                if number_test > num_images_test_total:
                    break
            print(len(giou_test_lst))
            test_iou_avg = sum(giou_test_lst) / len(giou_test_lst)
            print(test_iou_avg)
            if len(giou_test_lst_thre) > 0:
                test_iou_avg_thre = sum(giou_test_lst_thre) / len(giou_test_lst_thre)
                print(test_iou_avg_thre)

            print('---------------------------------------------------------------------------------------------------')
            print('---------------------------------------------------------------------------------------------------')
            print('---------------------------------------------------------------------------------------------------')




        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', model_out_cap['reward'].mean(), iteration)

            loss_history[iteration] = train_loss if not sc_flag else model_out_cap['reward'].mean()
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # update infos
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
               # eval model
            eval_kwargs = {'split': opt.split,                  ##### 'val'
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))

            # Dump miscalleous informations
            infos['best_val_score'] = best_val_score
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history

            save_checkpoint(model, infos, optimizer, histories)

            if opt.save_history_ckpt:
                save_checkpoint(model, infos, optimizer, append=str(iteration))

            #if best_flag:
            save_checkpoint(model, infos, optimizer, append='best')

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break



opt = opts.parse_opt()
train(opt)
