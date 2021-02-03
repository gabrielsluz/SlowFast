#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import math
import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper

from .build import MODEL_REGISTRY


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
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def forward(self, x):
        x = torch.flatten(x[0], start_dim=1)
        return self.lin(x)