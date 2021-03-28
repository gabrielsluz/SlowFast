import sys
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from slowfast.datasets.clevrer_res_monet import Clevrer_res_monet
from slowfast.models.mac_monet_v3 import MACNetwork
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
python3 clevrer_dev/mac/check_answers.py \
  --cfg clevrer_dev/mac/mac.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.DATASET Clevrer_res_monet \
  MAC.DIM 512 \
  MAC.MAX_STEPS 12 \
  MAC.DROPOUT 0.3 \
  WORD_EMB.EMB_DIM 300 \
  TRAIN.BATCH_SIZE 64 \
  LOG_PERIOD 200 \
  TRAIN.EVAL_PERIOD 3 \
  TRAIN.CHECKPOINT_FILE_PATH  ./checkpoint_mac/checkpoint_21_best.model \
  SOLVER.BASE_LR 1e-4 \
  NUM_GPUS 1 \
  DATA_LOADER.NUM_WORKERS 8\
  SOLVER.MAX_EPOCH 60

"""

logger = logging.get_logger(__name__)


def get_answers(epoch, valid_set):
    dataset = iter(valid_set)

    net.train(False)
    des_correct = {}
    des_wrong = {}
    anse_correct = {}
    anse_wrong = {}
    q_type_correct = {}
    q_type_total = {}
    correct_cnt = 0.0
    total_cnt = 0.0
    with torch.no_grad():
        for sampled_batch in tqdm(dataset):
            video_ft = sampled_batch['res_ft']
            question = sampled_batch['question_dict']['question']
            answer = sampled_batch['question_dict']['ans']
            q_len = sampled_batch['question_dict']['len']
            q_types = sampled_batch['question_dict']['question_type']
            video_ft['res_ft'] = video_ft['res_ft'].to(device)
            video_ft['monet_ft'] = video_ft['monet_ft'].to(device)
            question, answer, q_len = (
                question.to(device),
                answer.to(device),
                q_len.to(device)
            )

            output = net(video_ft, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, q_t in zip(correct, q_types):
                if not q_t in q_type_correct:
                    q_type_correct[q_t] = 0
                if not q_t in q_type_total:
                    q_type_total[q_t] = 0
                if c:
                    correct_cnt += 1
                    q_type_correct[q_t] += 1
                total_cnt += 1
                q_type_total[q_t] += 1
            
            for c, ans in zip(correct, output.detach().argmax(1)):
                ans = ans.item()
                if not ans in des_correct:
                    des_correct[ans] = 0
                if not ans in des_wrong:
                    des_wrong[ans] = 0
                if c:
                    des_correct[ans] += 1
                else:
                    des_wrong[ans] += 1
            
            for c, ans in zip(correct, answer.detach()):
                ans = ans.item()
                if not ans in anse_correct:
                    anse_correct[ans] = 0
                if not ans in anse_wrong:
                    anse_wrong[ans] = 0
                if c:
                    anse_correct[ans] += 1
                else:
                    anse_wrong[ans] += 1

    with open('answers_dist.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        w.write(str(des_correct) + "\n")
        w.write(str(des_wrong) + "\n")
        w.write(str(anse_correct) + "\n")
        w.write(str(anse_wrong) + "\n")

    for k,v in q_type_total.items():
        print("Question type {} Acc: {:.5f}".format(k, q_type_correct[k]/v))
    print(
        'Avg Acc: {:.5f}'.format(
            correct_cnt / total_cnt
        )
    )

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    #Vocabs
    vocab = {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2, ' counter ': 78, 'what': 3, 'is': 4, 'the': 5, 'shape': 6, 'of': 7, 'object': 8, 'to': 9, 'collide': 10, 'with': 11, 'purple': 12, '?': 13, 'color': 14, 'first': 15, 'gray': 16, 'sphere': 17, 'material': 18, 'how': 19, 'many': 20, 'collisions': 21, 'happen': 22, 'after': 23, 'cube': 24, 'enters': 25, 'scene': 26, 'objects': 27, 'enter': 28, 'are': 29, 'there': 30, 'before': 31, 'stationary': 32, 'rubber': 33, 'metal': 34, 'that': 35, 'moving': 36, 'cubes': 37, 'when': 38, 'blue': 39, 'exits': 40, 'any': 41, 'brown': 42, 'which': 43, 'following': 44, 'responsible': 45, 'for': 46, 'collision': 47, 'between': 48, 'and': 49, 'presence': 50, "'s": 51, 'entering': 52, 'colliding': 53, 'event': 54, 'will': 55, 'if': 56, 'cylinder': 57, 'removed': 58, 'collides': 59, 'without': 60, ',': 61, 'not': 62, 'red': 63, 'spheres': 64, 'exit': 65, 'cylinders': 66, 'video': 67, 'begins': 68, 'ends': 69, 'next': 70, 'last': 71, 'yellow': 72, 'cyan': 73, 'entrance': 74, 'green': 75, 'second': 76, 'exiting': 77}
    ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
    n_words = len(vocab.keys())
    n_answers = 21

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.NUM_GPUS else 'cpu')

    net = MACNetwork(cfg).to(device)
    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        net.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH))


    #Dataloaders
    valid_dst = Clevrer_res_monet(cfg, "val")
    valid_set = DataLoader(
        valid_dst, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=False
    )
    get_answers(20, valid_set)
    
    valid_dst.close()
