#!/usr/bin/env python3
from slowfast.datasets.clevrer_video import Clevrer_video
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from slowfast.models.video_model_builder import SlowFast

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/feature_extraction/generate_features.py \
  --cfg clevrer_dev/feature_extraction/slowfast.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer_video(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

model = SlowFast(cfg)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(model(sample_batched[0]))
    break
