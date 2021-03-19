#!/usr/bin/env python3
from slowfast.datasets.clevrer_dual import Clevrer_des
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import get_cfg
"""
Prints information about the dataset for testing and debugging
"""

cfg = get_cfg()
cfg.DATA.PATH_TO_DATA_DIR = '/datasets/clevrer'
cfg.DATA.PATH_PREFIX = '/datasets/clevrer'
cfg.DATA.RESIZE_H= 224
cfg.DATA.RESIZE_W= 224
cfg.DATA.NUM_FRAMES= 4
cfg.DATA.SAMPLING_RATE= 8



dataset = Clevrer_des(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

if len(dataset) > 5:
    max_i = 5
else:
    max_i = len(dataset)
print("First {} items".format(max_i))
for i in range(max_i):
    print(dataset.get_video_info(i))

# tensor_image = dataset[0][0][0].permute(1,2,0)
# plt.imshow(tensor_image)
# plt.savefig('sample_frame.png')

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0)
print("Vocab = {}".format(dataset.vocab))
print("Ans_vocab = {}".format(dataset.ans_vocab))
for i_batch, sample_batched in enumerate(dataloader):
    index = sample_batched['index'].item()
    print("Video info = {}".format(dataset.get_video_info(index)))
    print(sample_batched['frames'].size())
    print(sample_batched['question_dict']['question'])
    print(sample_batched['question_dict']['ans'])
    print(sample_batched['question_dict']['len'])
    print(sample_batched['index'])

    # for i_frame in range(sample_batched['frames'].size()[1]):
    #     plt.imshow(sample_batched['frames'][0][i_frame].permute(1,2,0))
    #     plt.savefig('./clevrer_dev/tests/sample_frame{}.png'.format(i_frame))
    # break

for i in range(1000):
    print(dataset._dataset[i])