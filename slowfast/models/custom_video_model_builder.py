#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import math
import torch
import torch.nn as nn

from .build import MODEL_REGISTRY

from .monet import Monet


@MODEL_REGISTRY.register()
class Linear(nn.Module):
    """
    Simple linear Neural Network for testing
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(Linear, self).__init__()
        num_classes=cfg.MODEL.NUM_CLASSES
        self.lin = nn.Linear(224*224*3, num_classes) #Requires TRAIN_CROP_SIZE: 224 TEST_CROP_SIZE: 224

    def forward(self, x):
        x = torch.flatten(x[0], start_dim=1)
        return self.lin(x)

# License: MIT
# Author: Karl Stelzner
from collections import namedtuple

config_options = [
    # Training config
    #'vis_every',  # Visualize progress every X iterations
    #'batch_size',
    #'num_epochs',
    #'load_parameters',  # Load parameters from checkpoint
    #'checkpoint_file',  # File for loading/storing checkpoints
    #'data_dir',  # Directory for the training data
    #'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
]

MonetConfig = namedtuple('MonetConfig', config_options)

@MODEL_REGISTRY.register()
class MonetModel(Monet):
    """
    Class used as an interface from PySlowFast with a MONet implementation in monet.py
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        clevr_conf = MonetConfig(num_slots=cfg.MONET.NUM_SLOTS,
                           num_blocks=6,
                           channel_base=64,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                          )
        height = cfg.DATA.RESIZE_H
        width = cfg.DATA.RESIZE_W
        super(MonetModel, self).__init__(clevr_conf, height, width)