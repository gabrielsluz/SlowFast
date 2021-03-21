"""
Implementation taken from https://github.com/rosinality/mac-network-pytorch
It was modified to use video input (SlowFast network features).

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
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

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


class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, embed_hidden=300,
                max_step=12, self_attention=False, memory_gate=False,
                classes=21, dropout=0.15):
        super().__init__()

        # self.conv = nn.Sequential(nn.Conv2d(1024, dim, 3, padding=1),
        #                         nn.ELU(),
        #                         nn.Conv2d(dim, dim, 3, padding=1),
        #                         nn.ELU(),
        #                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.res_proj = nn.Linear(2048, dim)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)


        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU(),
                                        linear(dim, classes))

        self.mc_pred_head = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU(),
                                        linear(dim, 4))

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        #kaiming_uniform_(self.conv[0].weight)
        #self.conv[0].bias.data.zero_()
        #kaiming_uniform_(self.conv[2].weight)
        #self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, video, question, question_len, is_des, dropout=0.15):
        b_size = question.size(0)

        # cb_sz = video.size()
        # img = self.conv(video.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
        # #N*T x C x H x W => N x T x C x H x W => N x C x T x H x W
        # img = img.view(cb_sz[0], cb_sz[1], self.dim, 7, 7).permute(0,2,1,3,4)
        # img = img.reshape(b_size, self.dim, -1)
        cb_sz = video.size()
        img = self.res_proj(video.view(cb_sz[0]*cb_sz[1], cb_sz[2]))
        img = img.view(cb_sz[0], cb_sz[1], self.dim).permute(0,2,1)
        img = img.reshape(b_size, self.dim, -1)


        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                enforce_sorted = False, batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        memory = self.mac(lstm_out, h, img)

        out = torch.cat([memory, h], 1)

        if is_des:
            out = self.classifier(out)
        else:
            out = self.mc_pred_head(out)

        return out
