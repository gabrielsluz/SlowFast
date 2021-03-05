#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test the CLEVRER model.
Example:

----Clevrer dataset-----

python3 clevrer_dev/baselines/run_net.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  TRAIN.DATASET Clevrer_des \
  TRAIN.ONLY_DES True \
  TRAIN.ENABLE True \
  WORD_EMB.USE_PRETRAINED_EMB False \
  WORD_EMB.TRAINABLE True \
  WORD_EMB.GLOVE_PATH '/datasets/word_embs/glove.6B/glove.6B.50d.txt' \
  WORD_EMB.EMB_DIM 50 \
  NUM_GPUS 0 \
  LOG_PERIOD 1 \
  TRAIN.BATCH_SIZE 2 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.MAX_EPOCH 1

python3 clevrer_dev/baselines/run_net.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.DATASET Clevrer_des \
  TRAIN.ONLY_DES True \
  TRAIN.ENABLE False \
  WORD_EMB.USE_PRETRAINED_EMB False \
  WORD_EMB.TRAINABLE True \
  WORD_EMB.GLOVE_PATH '/datasets/word_embs/glove.6B/glove.6B.50d.txt' \
  WORD_EMB.EMB_DIM 16 \
  CLEVRERMAIN.LSTM_HID_DIM 64 \
  DATA.NUM_FRAMES 10 \
  DATA.SAMPLING_RATE 12 \
  TRAIN.BATCH_SIZE 3 \
  NUM_GPUS 1 \
  LOG_PERIOD 100 \
  TRAIN.EVAL_PERIOD 2 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.EPOCH_CYCLE 2.0 \
  SOLVER.LR_POLICY cosine \
  SOLVER.BASE_LR 0.001 \
  SOLVER.COSINE_END_LR 0.00005 \
  SOLVER.WEIGHT_DECAY 0.0 \
  SOLVER.OPTIMIZING_METHOD adam \
  SOLVER.MAX_EPOCH 2
  """
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

#from demo_net import demo
# from test_net import test
from train_net import train
from train_net_des import train_des, test_implementation_des
# from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    if cfg.TRAIN.ENABLE:
        if cfg.TRAIN.ONLY_DES:
            launch_job(cfg=cfg, init_method=args.init_method, func=train_des)
        else:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
    else:
        if cfg.TRAIN.ONLY_DES:
            launch_job(cfg=cfg, init_method=args.init_method, func=test_implementation_des)
        else:
            launch_job(cfg=cfg, init_method=args.init_method, func=test_implementation)


if __name__ == "__main__":
    main()
