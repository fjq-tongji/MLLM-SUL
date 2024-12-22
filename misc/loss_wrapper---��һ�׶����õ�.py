import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import fvcore
from fvcore.nn import parameter_count_table
import thop
from thop import profile

def compute_param(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            y = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels, att_masks)[0]
            #macs, params = profile(self.model, inputs=(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels, att_masks))
            #print(fc_feats.shape)
            #print(att_feats.shape)
            #print(word_feats.shape)
            #print(attr_feats.shape)
            #print(seg_feats.shape)
            #print(boxes_feats.shape)
            #print(labels.shape)
            #print(att_masks.shape)

            #print(parameter_count_table(self.model))
            #print(compute_param(self.model))

            #attn = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels, att_masks)[1]
            
            loss = self.crit(self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, labels[..., :-1], att_masks), labels[:,1:], masks[:,1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
