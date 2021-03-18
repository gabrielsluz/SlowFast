#!/usr/bin/env python3
from slowfast.datasets.clevrer_video import Clevrer_video
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from slowfast.models.video_model_builder import SlowFast
import slowfast.utils.checkpoint as cu

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/feature_extraction/generate_features.py \
  --cfg clevrer_dev/feature_extraction/slowfast.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.CHECKPOINT_FILE_PATH model_zoo_ck/SLOWFAST_4x16_R50.pkl \
  TRAIN.CHECKPOINT_TYPE caffe2
"""

def forward(self, x, bboxes=None):
    x = self.s1(x)
    x = self.s1_fuse(x)
    x = self.s2(x)
    x = self.s2_fuse(x)
    for pathway in range(self.num_pathways):
        pool = getattr(self, "pathway{}_pool".format(pathway))
        x[pathway] = pool(x[pathway])
    x = self.s3(x)
    x = self.s3_fuse(x)
    x = self.s4(x)
    x = self.s4_fuse(x)
    x = self.s5(x)
    return x

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer_video(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

#model = SlowFast(cfg).cuda()
model = SlowFast(cfg)
cu.load_test_checkpoint(cfg, model)
model.eval()
model.forward = forward.__get__(model, SlowFast)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0)
for i_batch, sampled_batch in enumerate(dataloader):
    print(sampled_batch[0][0].size())
    out = model(sampled_batch[0])
    print(out[0].size())
    print(out[1].size())
    break
