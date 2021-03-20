import sys
import pickle
from collections import Counter

import torch
#from tqdm import tqdm
from torch.utils.data import DataLoader
from slowfast.config.defaults import get_cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cfg for dataset
cfg = get_cfg()
cfg.DATA.PATH_TO_DATA_DIR = '/datasets/clevrer'
cfg.DATA.PATH_PREFIX = '/datasets/clevrer'

checkpoint_path = sys.argv[1]
net = torch.load(checkpoint_path).to(device)

dataset = Clevrermac_des(cfg, "train")
train_loader = DataLoader(
    dataset, batch_size=5, num_workers=4, shuffle=False
)

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
    correct = output.detach().argmax(1) == answer.to(device)
    print("Output = {}".format(output))
    print("Answer = {}".format(answer))
    print("Correct = {}".format(correct))
    if cur_iter > 7:
        dataset.close()
        break
