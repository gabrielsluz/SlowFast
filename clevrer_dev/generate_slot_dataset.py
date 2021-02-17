"""
Generates a dataset by passing MONet through video clips.
Uses cfg to determine the parameters
Dataset format: 
A Python dictionary with video_path as key and a torch tensor as value
Tensor: (Num_frames * Num_slots) x Slot_dim
It should be used with clevrer json file

Example:
python3 clevrer_dev/generate_slot_dataset.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  CLEVRERMAIN.SLOT_DATASET_PATH /datasets/slot_dataset/slot_dataset.pyth \
  NUM_GPUS 0 \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth

Or:
python3 clevrer_dev/generate_slot_dataset.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  CLEVRERMAIN.SLOT_DATASET_PATH /datasets/slot_dataset/slot_dataset.pyth \
  NUM_GPUS 1 \
  MONET.CHECKPOINT_LOAD ./monet_checkpoints/checkpoint_epoch_00140.pyth
"""

from slowfast.models.monet import Monet
from collections import namedtuple
import slowfast.utils.checkpoint as cu
from slowfast.datasets.clevrer import Clevrer

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

#Fetch dataset
args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

#Load model
config_options = [
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
]
MonetConfig = namedtuple('MonetConfig', config_options)
clevr_conf = MonetConfig(num_slots=cfg.MONET.NUM_SLOTS,
                        num_blocks=6,
                        channel_base=64,
                        bg_sigma=0.09,
                        fg_sigma=0.11,
                        )
model = Monet(clevr_conf, cfg.DATA.RESIZE_H, cfg.DATA.RESIZE_W)
cu.load_checkpoint(cfg.MONET.CHECKPOINT_LOAD, model, data_parallel=False)
if cfg.NUM_GPUS:
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

dataset = Clevrer(cfg, 'train')

slot_dataset = {}

for i in range(len(dataset)):
    sampled_item = dataset[i]
    frames = sampled_item['frames']
    index = sampled_item['index']
    if cfg.NUM_GPUS:
        cur_device = torch.cuda.current_device()
        frames = frames.cuda(device=cur_device)
    video_slots = model.return_means(frames)
    video_path = dataset.get_video_path(index)
    slot_dataset[video_path] = video_slots.cpu()
    break

torch.save(slot_dataset, cfg.CLEVRERMAIN.SLOT_DATASET_PATH)
    
    