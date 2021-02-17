"""
Generates a dataset by passing MONet through video clips.
Uses cfg to determine the parameters
Dataset format: 
Uses the same format as Clevrer, but with tensors instead of videos
torch.load grabs the tensor

Example:
python3 clevrer_dev/generate_slot_dataset.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  CLEVRERMAIN.SLOT_DATASET_PATH /datasets/slot_dataset/ \
  NUM_GPUS 0 \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth

Or:
python3 clevrer_dev/generate_slot_dataset.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  CLEVRERMAIN.SLOT_DATASET_PATH /datasets/slot_dataset/ \
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

from pathlib import Path
import os



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
model.eval()

"""
Train dataset
Directory organization:
video_train 
Sub_dirs:
  video_00000-01000  video_01000-02000  video_02000-03000  video_03000-04000  video_04000-05000  
  video_05000-06000  video_06000-07000  video_07000-08000  video_08000-09000  video_09000-10000
"""
sub_dirs = ["video_00000-01000", "video_01000-02000", "video_02000-03000", "video_03000-04000", "video_04000-05000",
  "video_05000-06000", "video_06000-07000", "video_07000-08000", "video_08000-09000", "video_09000-10000"]

video_train_path = os.path.join(cfg.CLEVRERMAIN.SLOT_DATASET_PATH ,"video_train")
Path(video_train_path).mkdir(parents=True, exist_ok=True)
for sub_path in sub_dirs:
  path = os.path.join(video_train_path, sub_path)
  Path(path).mkdir(parents=True, exist_ok=True)

dataset = Clevrer(cfg, 'train')

for i in range(len(dataset)):
    sampled_item = dataset[i]
    frames = sampled_item['frames']
    index = sampled_item['index']
    if cfg.NUM_GPUS:
        cur_device = torch.cuda.current_device()
        frames = frames.cuda(device=cur_device)
    video_slots = model.return_means(frames)
    video_path = dataset.get_video_path(index)
    video_path_split = video_path.split('/')
    slot_path = os.path.join(video_train_path, video_path_split[-2], video_path_split[-1])
    torch.save(video_slots.cpu(), slot_path)
    print("Video {}, index {}".format(slot_path, i))
    
  
"""
Val dataset
Directory organization:
video_val
Sub_dirs:
  video_10000-11000  video_11000-12000  video_12000-13000  video_13000-14000  video_14000-15000
"""
sub_dirs = ["video_10000-11000", "video_11000-12000", "video_12000-13000", "video_13000-14000", "video_14000-15000"]

video_val_path = os.path.join(cfg.CLEVRERMAIN.SLOT_DATASET_PATH ,"video_val")
Path(video_val_path).mkdir(parents=True, exist_ok=True)
for sub_path in sub_dirs:
  path = os.path.join(video_val_path, sub_path)
  Path(path).mkdir(parents=True, exist_ok=True)

dataset = Clevrer(cfg, 'val')

for i in range(len(dataset)):
    sampled_item = dataset[i]
    frames = sampled_item['frames']
    index = sampled_item['index']
    if cfg.NUM_GPUS:
        cur_device = torch.cuda.current_device()
        frames = frames.cuda(device=cur_device)
    video_slots = model.return_means(frames)
    video_path = dataset.get_video_path(index)
    video_path_split = video_path.split('/')
    slot_path = os.path.join(video_val_path, video_path_split[-2], video_path_split[-1])
    torch.save(video_slots.cpu(), slot_path)
    print("Video {}, index {}".format(slot_path, i))