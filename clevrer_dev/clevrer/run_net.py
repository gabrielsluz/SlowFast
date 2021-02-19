#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test the CLEVRER model.
Example:

----Clevrer dataset-----
python3 clevrer_dev/clevrer/run_net.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth \
  NUM_GPUS 0 \
  LOG_PERIOD 1 \
  TRAIN.BATCH_SIZE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.MAX_EPOCH 1

python3 clevrer_dev/clevrer/run_net.py \
  --cfg clevrer_dev/clevrer/clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  MONET.CHECKPOINT_LOAD ./monet_checkpoints/checkpoint_epoch_00140.pyth \
  DATA.NUM_FRAMES 25 \
  DATA.SAMPLING_RATE 5 \
  CLEVRERMAIN.T_HID_DIM 1024 \
  NUM_GPUS 1 \
  LOG_PERIOD 25 \
  TRAIN.BATCH_SIZE 4 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.MAX_EPOCH 24

----Slot Clevrer-----
python3 clevrer_dev/clevrer/run_net.py \
  --cfg clevrer_dev/clevrer/slot_clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/slot_dataset \
  DATA.PATH_PREFIX /datasets/slot_dataset \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth \
  NUM_GPUS 0 \
  LOG_PERIOD 1 \
  TRAIN.BATCH_SIZE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.MAX_EPOCH 1

python3 clevrer_dev/clevrer/run_net.py \
  --cfg clevrer_dev/clevrer/slot_clevrer.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/slot_dataset \
  DATA.PATH_PREFIX /datasets/slot_dataset \
  MONET.CHECKPOINT_LOAD ./monet_checkpoints/checkpoint_epoch_00180.pyth \
  DATA.NUM_FRAMES 25 \
  DATA.SAMPLING_RATE 5 \
  NUM_GPUS 1 \
  LOG_PERIOD 100 \
  TRAIN.BATCH_SIZE 32 \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 10 \
  SOLVER.MAX_EPOCH 100
  """
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

#from demo_net import demo
# from test_net import test
from train_net import train
# from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    # if cfg.TEST.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    # if cfg.TENSORBOARD.ENABLE and (
    #     cfg.TENSORBOARD.MODEL_VIS.ENABLE
    #     or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    # ):
    #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    # if cfg.DEMO.ENABLE:
    #     demo(cfg)


if __name__ == "__main__":
    main()
