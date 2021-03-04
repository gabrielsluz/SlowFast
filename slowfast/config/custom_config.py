#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # Add your own customized configs.

    #Train and Val stats printing
    _C.TRAIN.TRAIN_STATS_FILE = "./train_stats.txt"
    #Train only des questions
    _C.TRAIN.ONLY_DES = False

    #Cycle cosine loss function
    _C.SOLVER.EPOCH_CYCLE = 20.0

    #MONET parameters
    _C.MONET = CfgNode()
    #Number of object slots
    _C.MONET.NUM_SLOTS = 8
    #Load monet checkpoint
    _C.MONET.CHECKPOINT_LOAD = "./checkpoints/monet.pyth"

    #Data params for resizing Height and Width
    _C.DATA.RESIZE_H = 64
    _C.DATA.RESIZE_W = 64

    #CLEVRERMain parameters
    _C.CLEVRERMAIN = CfgNode()
    #Transformer parameters:
    _C.CLEVRERMAIN.T_HEADS = 10
    _C.CLEVRERMAIN.T_LAYERS = 28
    _C.CLEVRERMAIN.T_HID_DIM = 1024
    _C.CLEVRERMAIN.T_DROPOUT = 0.1
    #Prediction Head hidden layer dimension
    _C.CLEVRERMAIN.PRED_HEAD_DIM = 128
    #MONet trainable:
    _C.CLEVRERMAIN.MONET_TRAINABLE = False
    #Slot dataset path
    _C.CLEVRERMAIN.SLOT_DATASET_PATH = '/datasets/slot_dataset/slot_dataset.pyth'

    #For word embeddings
    _C.WORD_EMB = CfgNode()
    _C.WORD_EMB.USE_PRETRAINED_EMB = False
    _C.WORD_EMB.TRAINABLE = True
    _C.WORD_EMB.GLOVE_PATH = '/datasets/word_embs/glove.6B.50d.txt'
    _C.WORD_EMB.EMB_DIM = 50

    

