#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.resnet import ResNet

import sys
import os

from tqdm import tqdm
import h5py

from slowfast.datasets.clevrer_video import Clevrer_video
import slowfast.utils.checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging


"""
Generates ResNet50 for the MAC model

Example:
python3 clevrer_dev/mac/generate_features.py \
  --cfg clevrer_dev/mac/mac.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer 
"""

#ResNet to extract features
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    return x


def gen_dataset(cfg, mode, root):
    #Generates one datasets for a certain split. => ResNet50
    #When using the generated file must indicate in which index the dataset starts to work
    #Train starts in 0
    #Val starts in 10000
    #Test starts in 15000
    dataset = Clevrer_video(cfg, mode)
    print("Dataset {} len = {}".format(mode, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    size = len(dataloader)
    batch_size = cfg.TRAIN.BATCH_SIZE

    h5_path = os.path.join(root, '{}_res50conv_features.hdf5'.format(mode))
    f_h5 = h5py.File(h5_path, 'w', libver='latest')
    d_set_h5 = f_h5.create_dataset('data', (size * batch_size, cfg.DATA.NUM_FRAMES, 1024, 14, 14),
                            dtype='f4', chunks=(16, cfg.DATA.NUM_FRAMES, 1024, 14, 14))
    # index_set_h5 = f_h5.create_dataset('indexes', (size * batch_size, 1),
    #                         dtype='f4')

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
            inputs = sampled_batch[0]
            indexes = sampled_batch[1]
            if cfg.NUM_GPUS:
                inputs = inputs.cuda(non_blocking=True)
            cb_sz = inputs.size()
            out = model(inputs.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
            out = out.view(cb_sz[0], cb_sz[1], 1024, 14, 14)
            d_set_h5[i_batch * batch_size:(i_batch + 1) * batch_size] = out.detach().cpu().numpy()
            #index_set_h5[i_batch * batch_size:(i_batch + 1) * batch_size] = indexes.detach().cpu().numpy().reshape(batch_size,1)
    f_h5.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    logger = logging.get_logger(__name__)
    logging.setup_logging(cfg.OUTPUT_DIR)
    use_gpu = cfg.NUM_GPUS > 0
    #Set model 
    model = torchvision.models.resnet50(pretrained=True, progress=True)
    if use_gpu:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
    model.forward = forward.__get__(model, ResNet)
    model.eval()

    #Proccess datasets
    root = cfg.DATA.PATH_TO_DATA_DIR
    gen_dataset(cfg, 'train', root)
    gen_dataset(cfg, 'val', root)
