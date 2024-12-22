# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
from .llama import ModelArgs, Transformer_llama
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, Blip2ForConditionalGeneration, Blip2Config
from transformers.models.bert.configuration_bert import BertConfig
from .qformer import BertLMHeadModel
import transformers
from transformers import T5Config, T5ForConditionalGeneration, Blip2QFormerModel, Blip2VisionModel


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MLP_bbox(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP_bbox, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers-1 else layer(x)  
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder_llama = decoder
        
    def forward(self, src, grid_feats, global_feats, semantic_feats, tgt, src_mask, tgt_mask):
        x, global_feats_ = self.encode(src, grid_feats, global_feats, semantic_feats, src_mask)
        x, reg = self.decode_llama(x, global_feats_, src_mask, tgt, tgt_mask)
        return x, reg
    
    def encode(self, src, grid_feats, global_feats, semantic_feats, src_mask):
        return self.encoder(src, grid_feats, global_feats, semantic_feats, src_mask)

    def decode_llama(self, memory, global_feats_, src_mask, tgt, tgt_mask):
        return self.decoder_llama(memory, global_feats_, src_mask, tgt, tgt_mask)



def init_vision_encoder(model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
    assert model_name in [
        "eva_clip_g",
        "eva2_clip_L",
        "clip_L",
    ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
    if model_name == "eva_clip_g":
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
    elif model_name == "clip_L":
        visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    self.vit_name = model_name
    return visual_encoder, ln_vision


def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
    local_path = '/data/fjq/blip-2/bert-base-uncased'
    encoder_config = BertConfig.from_pretrained(local_path, local_files_only=True)
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token

    Qformer = BertLMHeadModel(config=encoder_config)
    #.from_pretrained(local_path, config=encoder_config)

    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

    return Qformer, query_tokens







class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers_1 = clones(layer, 1)
        self.layers_2 = clones(layer, 1)
        self.layers_3 = clones(layer, 1)
        self.size = 512
        self.llama_dim = 4096
        self.dropout = 0.1
        self.query_len = 10
        self.num_query_token = 32

    def forward(self, att, grid_feats, global_feats, semantic_feats, mask):


        return grid_feats, grid_feats


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.dropout = dropout
        self.msa = MultiHeadedAttention(8, self.size)
        self.norm_1 = LayerNorm(self.size)
        self.dropout_1 = nn.Dropout(self.dropout)
    def forward(self, q, k, v, mask):
        x_yuan = q
        q = self.norm_1(q)
        x = self.msa(q, k, v, mask)
        x = self.dropout_1(x) + x_yuan

        return x



class EncoderBlock(nn.Module):
    def __init__(self, size, dropout):
        super(EncoderBlock, self).__init__()
        self.size = size
        self.dropout = dropout
        self.msa = MultiHeadedAttention(8, 512)
        self.ffn = PositionwiseFeedForward(self.size, self.size * 4)
        self.norm_1 = LayerNorm(self.size)
        self.norm_2 = LayerNorm(self.size)
        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        x_yuan = x
        src2 = self.norm_1(x)
        q = k = src2 #+ vl_pos
        src2 = self.msa(q, k, src2, mask)
        src = self.dropout_1(src2) + x_yuan

        x_yuan2 = src
        src2 = self.norm_2(src)
        src2 = self.ffn(src2)
        src = self.dropout_2(src2) + x_yuan2
        return src



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Decoder_llama(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.query_layer = 31
        self.query_len = 10            #####10
        self.dim = 4096
        self.size = 512
        self.L = 42                   ####描述的句子
        self.global_feats_len = 49
        self.num_total = 1 + self.global_feats_len + self.L

        self.t5_model = '/data/fjq/blip-2/blip2-flan-t5-xl-coco'
        self.config = Blip2Config.from_pretrained(self.t5_model)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.config.num_query_tokens, self.config.qformer_config.hidden_size))
        self.t5_model = Blip2ForConditionalGeneration.from_pretrained(
                self.t5_model, torch_dtype=torch.float16
        )
        self.qformer = Blip2QFormerModel(self.config.qformer_config)
        self.language_projection = nn.Linear(self.config.qformer_config.hidden_size, self.config.text_config.hidden_size)

        self.reg_token_1 = nn.Embedding(1, self.size)   ####(1,d)
        # self.fc = nn.Linear(self.dim, self.size, bias=False)
        # self.vl_pos_embed = nn.Embedding(self.num_total, self.size)   #(1+144+36,512)
        # self.reg_layer = clones(layer, 6)
        self.bbox_embed_1 = MLP_bbox(self.size, self.size, 4, 3)


    def forward(self, memory, global_feats_, src_mask, tgt, tgt_mask):       ##grid_feats=memory=global_feats_
        #print(self.t5_model)
        _bsz, seqlen = tgt.shape               ###########(b,100)

        #---------------------------------------------------------------------------------------------#QFormer
        #print(self.query_tokens.shape)
        query_tokens = self.query_tokens.expand(memory.shape[0], -1, -1)
        #print(query_tokens.shape)             ####(b,32,768)

        memory_mask = torch.ones(memory.size()[:-1], dtype=torch.long).to(memory.device)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            return_dict=True)
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        #print(language_model_inputs.shape)             ####(b,32,2048)
        #----------------------------------------------------------------------------------------------------------#

        tgt_embeds = self.t5_model.language_model.get_input_embeddings()(tgt)     ###tokens=(b,问题+答案=100,2048)
        tgt_embeds_answer = self.t5_model.language_model.get_input_embeddings()(tgt[:, 58:])

        memory_1 = torch.cat([language_model_inputs, tgt_embeds], dim=1)     ####(b,32+100=132,2048)
        outputs = self.t5_model.language_model(
            inputs_embeds=memory_1,
            decoder_inputs_embeds=tgt_embeds_answer,
            return_dict=True
        )
        outputs = outputs.logits
        print(outputs.shape)

        ########################################################################
        reg = self.reg_token_1.weight.unsqueeze(0).repeat(_bsz, 1, 1)   ####(b,1,512)
        # h_ = self.fc(h_[:, 58:, :])
        # h_ = torch.cat([reg, global_feats_, h_], dim=1)       ####(b,1+144+36,512)
        # vl_pos = self.vl_pos_embed.weight.unsqueeze(0).repeat(_bsz, 1, 1)   ####(b,1+144+36=181,512)
        # for layer in self.reg_layer:
        #    h_ = layer(h_, vl_pos, mask=None)                  ###(b,1+144+36,512)
        reg = self.bbox_embed_1(reg[:, 0, :]).sigmoid()           ###(b,512)  (b,4)
        ########################################################################

        return outputs, reg.float()

    def forward_inference(self, memory, tokens):
        _bsz, seqlen = tokens.shape               ###########(b,18)

        inputs_embeds = self.t5_model.language_model.get_input_embeddings()(tokens[:, 0:58])     ###tokens=(b,答案) -> (b,答案=42,512)
        inputs_embeds_prompt = inputs_embeds

        # ---------------------------------------------------------------------------------------------#QFormer
        query_tokens = self.query_tokens.expand(memory.shape[0], -1, -1)      ####(b,32,768)
        memory_mask = torch.ones(memory.size()[:-1], dtype=torch.long).to(memory.device)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            return_dict=True)
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)     ####(b,32,2048)
        # ----------------------------------------------------------------------------------------------------------#

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds_prompt], dim=1)    ###(b,32+问题=32+58,2048)

        atts_opt = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long).to(language_model_inputs.device)  ###(b,32,d)
        opt_attention_mask = torch.ones(inputs_embeds_prompt.size()[:-1], dtype=torch.long).to(inputs_embeds_prompt.device)  ###(b,问题,d)
        attention_mask = torch.cat([atts_opt, opt_attention_mask], dim=1)

        outputs = self.t5_model.language_model.generate(               ### (b,50)
            inputs_embeds=inputs_embeds, max_length=40, do_sample=True, top_p=0.75, temperature=0.1, repetition_penalty=1.0,
            length_penalty=1.0, attention_mask=attention_mask
        )

        return outputs



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

#####################################################################################################################################
#####################################################################################################################################

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class TransformerModel_blip2_t5(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder_llama(EncoderBlock(d_model, dropout)),
            #lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            #nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            )   #### d_model, tgt_vocab
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel_blip2_t5, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)  ### >6,m2transformer needs 3 at least
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)   ## 512
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        self.grid_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(2048),) if self.use_bn else ()) +
                (nn.Linear(2048, 1408),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(768),) if self.use_bn == 2 else ())))
        self.global_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(1536),) if self.use_bn else ()) +
                (nn.Linear(1536, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        # self.semantic_embed = nn.Sequential(*(
        #         ((nn.BatchNorm1d(4096),) if self.use_bn else ()) +
        #         (nn.Linear(4096, self.d_model),
        #          nn.ReLU(),
        #          nn.Dropout(self.drop_prob_lm)) +
        #         ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        #delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        self.word_embedding = Embeddings(self.d_model, tgt_vocab)
        # model_args: ModelArgs = ModelArgs()
        # model_args.vocab_size = 32000  ####self.tokenizer.n_words
        # self.llama = Transformer_llama(model_args)

        self.linear_fc = nn.Linear(2048, self.d_model, bias=False)  ##2048

        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

        self.get_trainable_params()

    def get_trainable_params(self):
        for name, para in self.named_parameters():
            para.requires_grad = True

        train_param_name = ['decoder_llama.t5_model'
                            ]
        for name, para in self.named_parameters():
            #print(name)
            for train_param in train_param_name:
                if train_param in name:
                    para.data = para.data.float()
                    para.requires_grad = True


    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, grid_feats, global_feats,semantic_feats, att_masks):
        
        #print('(((((((((((((((((((((((((((')
        #print(att_feats.shape)
        att_feats, grid_feats, global_feats, semantic_feats, seq, att_masks, seq_mask = \
            self._prepare_feature_forward(att_feats, grid_feats, global_feats, semantic_feats, att_masks)
        #print(att_feats.shape)
        memory, global_feats_ = self.model.encode(att_feats, grid_feats, global_feats, semantic_feats, att_masks)

        return fc_feats[...,:1], att_feats[...,:1], memory, att_masks   ##[...,:1]

    def _prepare_feature_forward(self, att_feats, grid_feats, global_feats, semantic_feats, att_masks=None, seq=None):
        att_masks_ = att_masks

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        grid_feats = pack_wrapper(self.grid_embed, grid_feats, att_masks=None)
        global_feats = pack_wrapper(self.global_embed, global_feats, att_masks=None)
        # semantic_feats = self.llama.tok_embeddings(semantic_feats)
        # semantic_feats = pack_wrapper(self.semantic_embed, semantic_feats, att_masks=None)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != 0) & (seq.data != 0)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]    ########################################  1
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )

        else:
            seq_mask = None

        return att_feats, grid_feats, global_feats, semantic_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, grid_feats, global_feats, semantic_feats, seq, att_masks=None):   ## word_feats
        #print(fc_feats.shape)         ###(5,2048)

        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])

        att_feats, grid_feats, global_feats, semantic_feats, seq, att_masks, seq_mask = \
            self._prepare_feature_forward(att_feats, grid_feats, global_feats, semantic_feats, att_masks, seq)

        out, reg = self.model(att_feats, grid_feats, global_feats, semantic_feats, seq, att_masks, seq_mask)

        return out, reg

    def core(self, it, fc_feats_ph, att_feats_ph, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, reg = self.model.decode_llama(memory, mask,                       ####self.model.decoder
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
