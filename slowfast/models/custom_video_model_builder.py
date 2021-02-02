#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import math
import torch
import torch.nn as nn

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

    def forward(self, x):
        print(len(x))
        x = torch.flatten(x[0], start_dim=1)
        return self.lin(x)