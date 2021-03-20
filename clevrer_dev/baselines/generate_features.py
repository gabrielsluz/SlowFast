#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import torchvision

import sys
import os

from tqdm import tqdm
import h5py

from slowfast.datasets.clevrer_video import Clevrer_video
import slowfast.utils.checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging


"""
Generates ResNet50 features for CNN_LSTM_Pretrained type architectures
We use a pre-trained ResNet-50 (He et al., 2016) to extract features from the video frames. 
We use the 2,048-dimensional pool5 layer output for CNN-based methods 
We uniformly sample 25 frames for each video as input.

Example:
python3 clevrer_dev/baselines/generate_features.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer 
"""

#ResNet to extract features from pool5
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    print(x.size())
    x = self.layer4(x)
    print(x.size())
    x = self.avgpool(x)
    print(x.size())
    x = torch.flatten(x, 1)
    print(x.size())
    #x = self.fc(x)
    return x


def gen_dataset(cfg, mode, root):
    #Generates one datasets for a certain split. => ResNet50 features pool5
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

    #h5py slow and fast datasets
    #Slow
    h5_path = os.path.join(root, '{}_res50_features.hdf5'.format(mode))
    f_h5 = h5py.File(h5_path, 'w', libver='latest')
    d_set_h5 = f_h5.create_dataset('data', (size * batch_size, 2048),
                            dtype='f4')

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
            inputs = sampled_batch[0]
            if cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            out = model(inputs)
            d_set_h5[i_batch * batch_size:(i_batch + 1) * batch_size] = out.detach().cpu().numpy()
    f_h5.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    logger = logging.get_logger(__name__)
    logging.setup_logging(cfg.OUTPUT_DIR)
    use_gpu = cfg.NUM_GPUS > 0
    #Set model 
    model = torchvision.models.resnet50(pretrained=True, progress=True, num_classes=1000)
    if use_gpu:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
    model.forward = forward.__get__(model, ResNet)
    model.eval()

    #Proccess datasets
    root = cfg.DATA.PATH_TO_DATA_DIR
    gen_dataset(cfg, 'train', root)
    gen_dataset(cfg, 'val', root)
