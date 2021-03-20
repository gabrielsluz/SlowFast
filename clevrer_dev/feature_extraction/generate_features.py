#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader

import sys
import os

from tqdm import tqdm
import h5py

from slowfast.datasets.clevrer_video import Clevrer_video
from slowfast.models.video_model_builder import SlowFast
import slowfast.utils.checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging


"""
Generates SlowFast features

Example:
python3 clevrer_dev/feature_extraction/generate_features.py \
  --cfg clevrer_dev/feature_extraction/slowfast.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.CHECKPOINT_FILE_PATH model_zoo_ck/SLOWFAST_4x16_R50.pkl \
  TRAIN.CHECKPOINT_TYPE caffe2
"""

#SlowFast feature extraction from almost last layer
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
    #x = self.s5(x)
    return x


def gen_dataset(cfg, mode, root):
    #Generates two datasets for a certain split. => Slow and fast features
    #When using the generated file must indicate in which index the dataset starts to work
    #torch.Size([50, 1280, 4, 14, 14]) torch.Size([50, 128, 32, 14, 14])
    #Train starts in 0
    #Val starts in 10000
    #Test starts in 15000
    dataset = Clevrer_video(cfg, mode)
    print("Dataset {} len = {}".format(mode, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    size = len(dataloader)
    batch_size = cfg.TRAIN.BATCH_SIZE

    #h5py slow and fast datasets
    #Slow
    slow_path = os.path.join(root, '{}_slow_features.hdf5'.format(mode))
    slow_h5 = h5py.File(slow_path, 'w', libver='latest')
    slow_dset = slow_h5.create_dataset('data', (size * batch_size, 1280, 4, 14, 14),
                            dtype='f4')
    #Fast
    fast_path = os.path.join(root, '{}_fast_features.hdf5'.format(mode))
    fast_h5 = h5py.File(fast_path, 'w', libver='latest')
    fast_dset = fast_h5.create_dataset('data', (size * batch_size, 128, 32, 14, 14),
                            dtype='f4')

    with torch.no_grad():
        for i, sampled_batch in tqdm(enumerate(dataloader)):
            inputs = sampled_batch[0]
            if cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            out = model(inputs)
            slow_ft = out[0].detach().cpu().numpy()
            fast_ft = out[1].detach().cpu().numpy() 
            slow_dset[i * batch_size:(i + 1) * batch_size] = slow_ft
            fast_dset[i * batch_size:(i + 1) * batch_size] = fast_ft

    slow_h5.close()
    fast_h5.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    logger = logging.get_logger(__name__)
    logging.setup_logging(cfg.OUTPUT_DIR)
    use_gpu = cfg.NUM_GPUS > 0
    #Set model 
    model = SlowFast(cfg)
    if use_gpu:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
    cu.load_test_checkpoint(cfg, model)
    model.forward = forward.__get__(model, SlowFast)
    model.eval()

    #Proccess datasets
    root = cfg.DATA.PATH_TO_DATA_DIR
    gen_dataset(cfg, 'train', root)
    gen_dataset(cfg, 'val', root)