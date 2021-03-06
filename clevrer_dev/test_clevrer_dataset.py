#!/usr/bin/env python3
from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/test_clevrer_dataset.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

if len(dataset) > 5:
    max_i = 5
else:
    max_i = len(dataset)
print("First {} items".format(max_i))
for i in range(max_i):
    print(dataset.get_video_info(i))
print(dataset.vocab)
# tensor_image = dataset[0][0][0].permute(1,2,0)
# plt.imshow(tensor_image)
# plt.savefig('sample_frame.png')

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['frames'].size())
    print(sample_batched['question_dict']['des_q'].size())
    print(sample_batched['question_dict']['des_ans'].size())
    print(sample_batched['question_dict']['mc_q'].size())
    print(sample_batched['question_dict']['mc_ans'].size())
    print(sample_batched['index'].size())

    for i_frame in range(sample_batched['frames'].size()[1]):
        plt.imshow(sample_batched['frames'][0][i_frame].permute(1,2,0))
        #plt.savefig('sample_frame{}.png'.format(i_frame))
    break