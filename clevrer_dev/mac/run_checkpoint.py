import sys
import pickle
from collections import Counter

import torch
import torch.nn as nn
#from tqdm import tqdm
from torch.utils.data import DataLoader
from slowfast.config.defaults import get_cfg
from slowfast.datasets.clevrer_mac import Clevrermac_des
from slowfast.models.mac import MACNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cfg for dataset
cfg = get_cfg()
cfg.DATA.PATH_TO_DATA_DIR = '/datasets/clevrer'
cfg.DATA.PATH_PREFIX = '/datasets/clevrer'

vocab = {' CLS ': 0, ' PAD ': 1, '|': 2, ' counter ': 78, 'what': 3, 'is': 4, 'the': 5, 'shape': 6, 'of': 7, 'object': 8, 'to': 9, 'collide': 10, 'with': 11, 'purple': 12, '?': 13, 'color': 14, 'first': 15, 'gray': 16, 'sphere': 17, 'material': 18, 'how': 19, 'many': 20, 'collisions': 21, 'happen': 22, 'after': 23, 'cube': 24, 'enters': 25, 'scene': 26, 'objects': 27, 'enter': 28, 'are': 29, 'there': 30, 'before': 31, 'stationary': 32, 'rubber': 33, 'metal': 34, 'that': 35, 'moving': 36, 'cubes': 37, 'when': 38, 'blue': 39, 'exits': 40, 'any': 41, 'brown': 42, 'which': 43, 'following': 44, 'responsible': 45, 'for': 46, 'collision': 47, 'between': 48, 'and': 49, 'presence': 50, "'s": 51, 'entering': 52, 'colliding': 53, 'event': 54, 'will': 55, 'if': 56, 'cylinder': 57, 'removed': 58, 'collides': 59, 'without': 60, ',': 61, 'not': 62, 'red': 63, 'spheres': 64, 'exit': 65, 'cylinders': 66, 'video': 67, 'begins': 68, 'ends': 69, 'next': 70, 'last': 71, 'yellow': 72, 'cyan': 73, 'entrance': 74, 'green': 75, 'second': 76, 'exiting': 77}
ans_vocab = {' counter ': 21, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'yes': 6, 'no': 7, 'rubber': 8, 'metal': 9, 'sphere': 10, 'cube': 11, 'cylinder': 12, 'gray': 13, 'brown': 14, 'green': 15, 'red': 16, 'blue': 17, 'purple': 18, 'yellow': 19, 'cyan': 20}
n_words = len(vocab.keys())
n_answers = 21
dim = 512
net = MACNetwork(n_words, dim).to(device)

checkpoint_path = sys.argv[1]
net.load_state_dict(torch.load(checkpoint_path))

dataset = Clevrermac_des(cfg, "train")
train_loader = DataLoader(
    dataset, batch_size=1, num_workers=4, shuffle=False
)
criterion = nn.CrossEntropyLoss()

for cur_iter, sampled_batch in enumerate(train_loader):
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
    output = net(slow_ft, fast_ft, question, q_len, True)
    loss = criterion(output, answer)
    correct = output.detach().argmax(1) == answer.to(device)
    print("Begin printing ----")
    print("Slow ft sum = {}".format(slow_ft.sum()))
    print("Fast ft sum = {}".format(fast_ft.sum()))
    print("Video info = {}".format(dataset.get_video_info(cur_iter)))
    print("Dataset entry = {}".format(dataset._dataset[cur_iter]))
    print("Output = {}".format(output))
    print("Output argmax = {}".format(output.detach().argmax(1)))
    print("Answer = {}".format(answer))
    print("Correct = {}".format(correct))
    print("Loss  = {}".format(loss))
    print("End printing ----")
    if cur_iter > 50:
        dataset.close()
        break
