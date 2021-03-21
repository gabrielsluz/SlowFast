"""
Implementation of MAC taken and modified from https://github.com/rosinality/mac-network-pytorch

MIT License

Copyright (c) 2018 Kim Seonghyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
from transformers import BertModel

from .build import MODEL_REGISTRY

#MAC network

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory



@MODEL_REGISTRY.register()
class BERT_CNN_MAC(nn.Module):
    """
    Implemetation of the model BERT+CNN+MAC model for Clevrer
    Receives ResNet101 layer3 features, pass them through a small CNN
    Pass question through BERT
    Use MAC.
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        print("BERT_CNN_MAC model")
        super(BERT_CNN_MAC, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #MAC params
        classes = 21
        dim = cfg.MAC.DIM
        max_step = cfg.MAC.MAX_STEPS
        self_attention = False
        memory_gate = False
        dropout = cfg.MAC.DROPOUT
        #BERT
        self.BERT = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hid_dim = self.BERT.config.hidden_size
        self.bert_proj = nn.Linear(self.bert_hid_dim, dim)
        self.bert_cls_proj = nn.Linear(self.bert_hid_dim, dim)
        #CONV
        self.conv = nn.Sequential(nn.Conv2d(1024, dim, kernel_size=3, stride=3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        #Generates => N x dim x 3 x 3
        #MAC UNIT
        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)

        #MAC final prediciotn head
        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU(),
                                        linear(dim, classes))
        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)


    def forward(self, video, question, question_len, dropout=0.15):
        cb_sz = video.size()
        frame_encs = self.conv(video.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
        #N*T x C x H x W => N x T x C x H x W
        frame_encs = frame_encs.view(cb_sz[0], cb_sz[1], self.dim, 3, 3)
        frame_encs = frame_encs.permute(0,2,1,3,4).contiguous().view(cb_sz[0], self.dim, -1)

        bert_out = self.BERT(input_ids=question['input_ids'],
                            attention_mask=question['attention_mask'],
                            token_type_ids=question['token_type_ids'])
        q_encs = self.bert_proj(bert_out.last_hidden_state)
        h = self.bert_cls_proj(bert_out.pooler_output)

        memory = self.mac(q_encs, h, frame_encs)

        out = torch.cat([memory, h], 1)

        out = self.classifier(out)

        return out








# @MODEL_REGISTRY.register()
# class BERT_CNN_LSTM(nn.Module):
#     """
#     Implemetation of a baseline BERT+CNN+LSTM model for Clevrer
#     Receives ResNet101 layer3 features, pass them through a small CNN
#     Pass question through BERT
#     Pass word_encs and frame_encs through a LSTM and Linear head to generate output
#     """

#     def init_params(self, layer):
#         if type(layer) == nn.Embedding:
#             nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
#             nn.init.zeros_(layer.weight[layer.padding_idx])
#         elif type(layer) == nn.Linear:
#             nn.init.xavier_normal_(layer.weight)
#             # nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#             nn.init.normal_(layer.bias)
#         elif type(layer) == nn.Conv2d:
#             nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
#             nn.init.normal_(layer.bias)

#     def __init__(self, cfg):
#         """
#         The `__init__` method of any subclass should also contain these
#             arguments.

#         Args:
#             cfg (CfgNode): model building configs, details are in the
#                 comments of the config file.
#         """
#         print("BERT_CNN_LSTM model")
#         super(BERT_CNN_LSTM, self).__init__()
#         #CUDA
#         self.num_gpus = cfg.NUM_GPUS
#         self.ans_vocab_len = 21
#         #BERT
#         self.BERT = BertModel.from_pretrained('bert-base-uncased')
#         self.bert_hid_dim = self.BERT.config.hidden_size
#         #Conv
#         c_dim = 256
#         self.conv = nn.Sequential(nn.Conv2d(1024, c_dim, 3, padding=1),
#                                 nn.ELU(),
#                                 nn.Conv2d(c_dim, c_dim, 3, padding=1),
#                                 nn.ELU(),
#                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#Use a better pool
#         self.proj_frame_enc = nn.Linear(c_dim*7*7, self.bert_hid_dim)
#         #LSTM
#         self.hid_st_dim = cfg.CLEVRERMAIN.LSTM_HID_DIM
#         self.num_layers = 2
#         self.num_directions = 2
#         self.LSTM = torch.nn.LSTM(
#             input_size=self.bert_hid_dim, hidden_size=self.hid_st_dim, num_layers=self.num_layers,
#             bias=True, batch_first=True, dropout=cfg.CLEVRERMAIN.T_DROPOUT, bidirectional=True
#         )
#         #Prediction head MLP
#         ph_input_dim = self.hid_st_dim*2
#         self.des_pred_head = nn.Linear(ph_input_dim, self.ans_vocab_len)
#         #Init parameters *embed layer is initialized above
#         self.des_pred_head.apply(self.init_params)
#         self.conv.apply(self.init_params)

#     def forward(self, clips_b, question_b, is_des_q):
#         """
#         Receives a batch of clips and questions:
#                 clips_b (tensor): the frames of sampled from the video. The dimension
#                     is `batch_size` x `num frames` x `channel` x `height` x `width`.
#                 question_b (tensor): The dimension is
#                     `batch_size` x 'max sequence length'
#                 is_des_q (bool): Indicates if is descriptive question or multiple choice
#         """
#         #Receives a batch of frames
#         cb_sz = clips_b.size()
#         frame_encs = self.conv(video.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
#         #N*T x C x H x W => N x T x C x H x W
#         frame_encs = frame_encs.view(cb_sz[0], cb_sz[1], self.dim, 7, 7)
#         img = img.reshape(b_size, self.dim, -1)
#         frame_encs
#         #BERT pooler_output or last_hidden_state
#         bert_out = self.BERT(input_ids=question_b['input_ids'],
#                             attention_mask=question_b['attention_mask'],
#                             token_type_ids=question_b['token_type_ids'])
#         q_encs = bert_out.last_hidden_state
#         x = torch.cat((q_encs, frame_encs), dim=1)
#         if is_des_q:
#             return self.des_pred_head(x)
#         else:
#             return self.mc_pred_head(x)