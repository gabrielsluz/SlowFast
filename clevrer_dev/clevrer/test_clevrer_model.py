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
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)

vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()

model = ClevrerMain(cfg, vocab_len, ans_vocab_len)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['frames'].size())
    print(sample_batched['question_dict']['des_q'].size())
    print(sample_batched['question_dict']['des_ans'].size())
    print(sample_batched['question_dict']['mc_q'].size())
    print(sample_batched['question_dict']['mc_ans'].size())
    print(sample_batched['index'].size())
    break