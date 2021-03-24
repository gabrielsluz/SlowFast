#!/usr/bin/env python3
from slowfast.datasets.clevrer_sf import Clevrer_sf
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/feature_extraction/test_clevrer_dataset.py \
  --cfg clevrer_dev/feature_extraction/slowfast.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer_sf(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)
print("Vocab = {}".format(dataset.vocab))
print("Ans_vocab = {}".format(dataset.ans_vocab))

for i in range(10):
    print(dataset[i]['res_ft']['slow_ft'].sum())
    print(dataset[i]['res_ft']['fast_ft'].sum())

for i_batch, sampled_batch in enumerate(dataloader):
    index = sampled_batch['index'][0].item()
    print("Video info = {}".format(dataset.get_video_info(index)))
    print(sampled_batch['res_ft'].size())
    print(sampled_batch['question_dict']['question'])
    print(sampled_batch['question_dict']['attention_mask'])
    print(sampled_batch['question_dict']['ans'])
    print(sampled_batch['question_dict']['len'])

    break

dataset = Clevrer_sf(cfg, 'val')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)
print("Vocab = {}".format(dataset.vocab))
print("Ans_vocab = {}".format(dataset.ans_vocab))

for i_batch, sampled_batch in enumerate(dataloader):
    index = sampled_batch['index'][0].item()
    print("Video info = {}".format(dataset.get_video_info(index)))
    print(sampled_batch['res_ft']['slow_ft'].size())
    print(sampled_batch['res_ft']['fast_ft'].size())
    print(sampled_batch['question_dict']['question'])
    print(sampled_batch['question_dict']['attention_mask'])
    print(sampled_batch['question_dict']['ans'])
    print(sampled_batch['question_dict']['len'])

    break

for i in range(10):
    print(dataset[i]['res_ft']['slow_ft'].sum())
    print(dataset[i]['res_ft']['fast_ft'].sum())