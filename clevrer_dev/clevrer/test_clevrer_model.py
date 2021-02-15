#!/usr/bin/env python3
from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from slowfast.models.clevrer_model import ClevrerMain

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/clevrer/test_clevrer_model.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth

python3 clevrer_dev/clevrer/test_clevrer_model.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  MONET.CHECKPOINT_LOAD ./checkpoints/checkpoint_epoch_00140.pyth
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
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0)

vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()

model = ClevrerMain(cfg, vocab_len, ans_vocab_len)

print("Number of parameters = {}".format(count_parameters(model)))

for i_batch, sampled_batch in enumerate(dataloader):
    print("Batch info:")
    print(sampled_batch['frames'].size())
    print(sampled_batch['question_dict']['des_q'].size())
    print(sampled_batch['question_dict']['des_ans'].size())
    print(sampled_batch['question_dict']['mc_q'].size())
    print(sampled_batch['question_dict']['mc_ans'].size())
    print(sampled_batch['index'].size())

    print("Passing through model")
    model(sampled_batch['frames'], sampled_batch['question_dict']['des_q'])

    break