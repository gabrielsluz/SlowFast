#!/usr/bin/env python3
from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from slowfast.models.cnn_models import CNN_MLP

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging


"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/baselines/test_cnn_model.py \
  --cfg clevrer_dev/baselines/cnn_mlp.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  DATA.NUM_FRAMES 1 \
  NUM_GPUS 0

python3 clevrer_dev/baselines/test_cnn_model.py \
  --cfg clevrer_dev/baselines/cnn_mlp.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer
"""

#https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True, num_workers=0)

vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()

model = CNN_MLP(cfg, vocab_len, ans_vocab_len)
if cfg.NUM_GPUS:
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

print("Number of parameters = {}".format(count_parameters(model)))


for i_batch, sampled_batch in enumerate(dataloader):
    print("Batch info:")
    print(sampled_batch['frames'].size())
    print(sampled_batch['question_dict']['des_q'].size())
    print(sampled_batch['question_dict']['des_ans'].size())
    print(sampled_batch['question_dict']['mc_q'].size())
    print(sampled_batch['question_dict']['mc_ans'].size())
    print(sampled_batch['index'].size())

    frames = sampled_batch['frames']
    des_q = sampled_batch['question_dict']['des_q']

    print("Passing through model")
    if cfg.NUM_GPUS:
        cur_device = torch.cuda.current_device()
        frames = frames.cuda(device=cur_device)
        des_q = des_q.cuda(device=cur_device)
    output = model(frames, des_q, True)
    print(output.size())
    break
