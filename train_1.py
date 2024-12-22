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
from dataloader_1 import *
import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
#from misc.loss_wrapper import LossWrapper
import fvcore
from fvcore.nn import parameter_count_table
import copy


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    #del opt.vocab
    #dp_model = torch.nn.DataParallel(model)  ### dp_model
    # dp_model = model                           #####单个gpu
    #lw_model = LossWrapper(model, opt)  ##model
    # dp_lw_model = lw_model
    if torch.cuda.device_count() > 1:
        dp_lw_model = torch.nn.DataParallel(model).cuda()
    else:
        dp_lw_model = model.cuda()
#---------------------------------------------------------------------------------------------# 把LLAMA-7B的参数喂进到模型中
    device_0 = torch.device("cuda:0")
    checkpoint_llama = torch.load(opt.pretrained_llama, map_location=device_0)
    dp_lw_model.load_state_dict(checkpoint_llama, strict=False)                 ###model
    print('************')
    print('Pre-training LLAMA weights OK!')
    print('************')
# -----------------------------------------------------------------------------------------------------------------------#

    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    if opt.noamopt:
        assert opt.caption_model in ['transformer','aoa'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        #optimizer = utils.build_optimizer(model.parameters(), opt)
        model_parameters_filter = filter(lambda p: p.requires_grad, model.parameters())   ###289个参数不用训练
        optimizer = utils.build_optimizer(model_parameters_filter, opt)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))




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
        tmp = [data['fc_feats'], data['att_feats'], data['grid_feats'], data['global_feats'], data['tokens'], data['masks'], data['att_masks']]

        #print('#################masks###########')
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        fc_feats, att_feats, grid_feats, global_feats, tokens, masks, att_masks = tmp

        model_out_cap = dp_lw_model(fc_feats, att_feats, grid_feats, global_feats, tokens, att_masks)[0]
        #model_out_reg = dp_lw_model(fc_feats, att_feats, grid_feats, global_feats, tokens, att_masks)[1]
        ### tokens[..., :-1]

        #crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        #---------------------------------------------------------------------------------------------------------------#制作标签
        #---------------------------------------------------------------------------------------------------------------#
        tokens_target = copy.deepcopy(tokens)
        length = 42
        tokens_target[:, :length] = -1
        target_mask = tokens_target.ge(0)
        tokens_target[~target_mask] = 0         ###tokens_target(5,60)
        # --------------------------------------------------------------------------------------------------------------#
        # --------------------------------------------------------------------------------------------------------------#

        #loss = crit(model_out_cap[:, :-1], tokens_target[:, 1:], masks[:, 1:]).mean()     ###model_out_cap
        creterion = torch.nn.CrossEntropyLoss(ignore_index=0)    #### 0被忽略
        out_ = model_out_cap[:, :-1]
        labels_ = tokens_target[:, 1:]
        loss_cap = creterion(out_.reshape(-1, 32000), labels_.flatten())        ####描述损失
        #loss_reg =  model_out_reg                                                            ####定位损失
        loss = loss_cap

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
            print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, model_out_cap['reward'].mean(), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

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
