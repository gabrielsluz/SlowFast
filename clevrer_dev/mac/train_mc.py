import sys
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from slowfast.datasets.clevrer_mac import Clevrermac_des
from slowfast.models.mac import MACNetwork
from slowfast.config.defaults import get_cfg

batch_size = 32
n_epoch = 20
dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#cfg for dataset
cfg = get_cfg()
cfg.DATA.PATH_TO_DATA_DIR = '/datasets/clevrer'
cfg.DATA.PATH_PREFIX = '/datasets/clevrer'


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch):
    clevr = Clevrermac_des(cfg, "train")
    train_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss_q = 0
    moving_loss_opt = 0

    net.train(True)
    for sampled_batch in pbar:
        slow_ft = sampled_batch['slow_ft']
        fast_ft = sampled_batch['fast_ft']
        question = sampled_batch['question_dict']['question']
        answer = sampled_batch['question_dict']['ans']
        q_len = sampled_batch['question_dict']['len']
        is_des = sampled_batch['question_dict']['is_des']
        slow_ft, fast_ft, question, answer, is_des = (
            slow_ft.to(device),
            fast_ft.to(device),
            question.to(device),
            answer.to(device),
            is_des.to(device)
        )

        net.zero_grad()
        output = net(slow_ft, fast_ft, question, q_len, is_des)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()

        diff_mc_ans = torch.abs(answer - (torch.sigmoid(output) >= 0.5).float()) #Errors
        mc_opt_err = 100 * torch.true_divide(diff_mc_ans.sum(), (4*question.size()[0]))
        mc_q_err = 100 * torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) != 0).float().sum(), question.size()[0])
        # Gather all the predictions across all the devices.
        if torch.cuda.is_available():
            mc_opt_err, mc_q_err  = du.all_reduce(
                [mc_opt_err, mc_q_err]
            )

        if moving_loss_q == 0:
            moving_loss_q = mc_q_err

        else:
            moving_loss_q = moving_loss_q * 0.99 + mc_q_err * 0.01

        if moving_loss_opt == 0:
            moving_loss_opt = mc_opt_err

        else:
            moving_loss_opt = moving_loss_opt * 0.99 + mc_opt_err * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; mc_q_err: {:.5f}; mc_opt_err: {:.5f}'.format(
                epoch + 1, loss.item(), moving_loss_q, moving_loss_opt
            )
        )

        accumulate(net_running, net)

    clevr.close()


def valid(epoch):
    clevr = Clevrermac_des(cfg, "val")
    valid_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4
    )
    dataset = iter(valid_set)

    net_running.train(False)
    num_mc_opt_mis = 0.0
    num_mc_q_mis = 0.0
    num_samples = 0.0
    with torch.no_grad():
        for sampled_batch in tqdm(dataset):
            slow_ft = sampled_batch['slow_ft']
            fast_ft = sampled_batch['fast_ft']
            question = sampled_batch['question_dict']['question']
            answer = sampled_batch['question_dict']['ans']
            q_len = sampled_batch['question_dict']['len']
            slow_ft, fast_ft, question, answer = (
                slow_ft.to(device),
                fast_ft.to(device),
                question.to(device),
                answer.to(device),
            )

            output = net_running(slow_ft, fast_ft, question, q_len)
            diff_mc_ans = torch.abs(answer - (torch.sigmoid(output) >= 0.5).float()) #Errors
            mc_opt_err = 100 * torch.true_divide(diff_mc_ans.sum(), (4*question.size()[0]))
            mc_q_err = 100 * torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) != 0).float().sum(), question.size()[0])
            # Gather all the predictions across all the devices.
            if torch.cuda.is_available():
                mc_opt_err, mc_q_err  = du.all_reduce(
                    [mc_opt_err, mc_q_err]
                )
            num_samples += question.size()[0]
            num_mc_opt_mis += mc_opt_err * question.size()[0]
            num_mc_q_mis += mc_q_err * question.size()[0]

    # with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
    #     for k, v in family_total.items():
    #         w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    print(
        'Avg mc_q_err: {:.5f} mc_opt_err: {:.5f}'.format(
            num_mc_q_mis / num_samples, num_mc_opt_mis / num_samples
        )
    )

    clevr.close()


if __name__ == '__main__':
    #Vocabs
    vocab = {' CLS ': 0, ' PAD ': 1, '|': 2, ' counter ': 74, 'what': 3, 'is': 4, 'the': 5, 'shape': 6, 'of': 7, 'object': 8, 'to': 9, 'collide': 10, 'with': 11, 'purple': 12, '?': 13, 'color': 14, 'first': 15, 'gray': 16, 'sphere': 17, 'material': 18, 'how': 19, 'many': 20, 'collisions': 21, 'happen': 22, 'after': 23, 'cube': 24, 'enters': 25, 'scene': 26, 'objects': 27, 'enter': 28, 'are': 29, 'there': 30, 'before': 31, 'stationary': 32, 'rubber': 33, 'metal': 34, 'that': 35, 'moving': 36, 'cubes': 37, 'when': 38, 'blue': 39, 'exits': 40, 'any': 41, 'brown': 42, 'which': 43, 'following': 44, 'responsible': 45, 'for': 46, 'collision': 47, 'between': 48, 'and': 49, "'s": 50, 'colliding': 51, 'event': 52, 'will': 53, 'if': 54, 'cylinder': 55, 'removed': 56, 'without': 57, ',': 58, 'not': 59, 'red': 60, 'spheres': 61, 'exit': 62, 'cylinders': 63, 'video': 64, 'begins': 65, 'ends': 66, 'next': 67, 'last': 68, 'yellow': 69, 'cyan': 70, 'green': 71, 'second': 72, 'exiting': 73}
    ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
    n_words = len(vocab.keys())
    n_answers = 21

    net = MACNetwork(n_words, dim).to(device)
    net_running = MACNetwork(n_words, dim).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        train(epoch)
        valid(epoch)

        with open(
            'checkpoint_mac/checkpoint_mc{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
        ) as f:
            torch.save(net_running.state_dict(), f)
