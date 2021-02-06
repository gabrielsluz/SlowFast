#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Add your own customized configs.

    #MONET parameters
    _C.MONET = CfgNode()

    #Number of object slots
    _C.MONET.NUM_SLOTS = 11

    #Data params for resizing Height and Width
    _C.DATA.RESIZE_H = 64
    _C.DATA.RESIZE_W = 64
