#!/usr/bin/env python3
from slowfast.datasets.clevrer_bert_resnet import Clevrerbert_resnet
import torch
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/bert_mac/test_dataset.py \
  --cfg clevrer_dev/bert_mac/bert_mac.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrerbert_resnet(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)

for i in range(10):
    print("Size = {}".format(dataset[i]['res_ft'].size()))
    print("Sum = {}".format(dataset[i]['res_ft'].sum()))
    print("Index dataset = {} Res_ft index = {}".format(dataset[i]['res_ft_index'],i))

for i_batch, sampled_batch in enumerate(dataloader):
    index = sampled_batch['index'][0].item()
    print("Video info = {}".format(dataset.get_video_info(index)))
    print("Video info = {}".format(dataset.get_video_info(sampled_batch['index'][1].item())))
    print(sampled_batch['res_ft'].size())
    print(sampled_batch['question_dict']['question'])
    print(sampled_batch['question_dict']['question_type'])
    print(sampled_batch['question_dict']['ans'])
    print(sampled_batch['index'], sampled_batch['res_ft_index'])

    break

dataset = Clevrerbert_resnet(cfg, 'val')
print("Dataset len = {}".format(len(dataset)))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=True, num_workers=0)

for i_batch, sampled_batch in enumerate(dataloader):
    index = sampled_batch['index'][0].item()
    print("Video info = {}".format(dataset.get_video_info(index)))
    print("Video info = {}".format(dataset.get_video_info(sampled_batch['index'][1].item())))
    print(sampled_batch['res_ft'].size())
    print(sampled_batch['question_dict']['question'])
    print(sampled_batch['question_dict']['question_type'])
    print(sampled_batch['question_dict']['ans'])
    print(sampled_batch['index'], sampled_batch['res_ft_index'])

    break
    
for i in range(10):
    print("Size = {}".format(dataset[i]['res_ft'].size()))
    print("Sum = {}".format(dataset[i]['res_ft'].sum()))
    print("Index dataset = {} Res_ft index = {}".format(dataset[i]['res_ft_index'],i))
