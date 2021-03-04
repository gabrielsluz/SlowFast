#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test the CLEVRER model.
Example:
----Clevrer dataset-----
python3 clevrer_dev/text_baseline/run_net.py \
  --cfg clevrer_dev/text_baseline/text_gru.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  TRAIN.DATASET Clevrertext_des \
  NUM_GPUS 0 \
  LOG_PERIOD 1 \
  TRAIN.BATCH_SIZE 3 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  WORD_EMB.USE_PRETRAINED_EMB False \
  WORD_EMB.TRAINABLE True \
  WORD_EMB.GLOVE_PATH '/datasets/word_embs/glove.6B/glove.6B.50d.txt' \
  WORD_EMB.EMB_DIM 50 \
  TRAIN.ONLY_DES True \
  SOLVER.MAX_EPOCH 2

python3 clevrer_dev/text_baseline/run_net.py \
  --cfg clevrer_dev/text_baseline/text_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  NUM_GPUS 1 \
  LOG_PERIOD 20 \
  TRAIN.BATCH_SIZE 20 \
  TRAIN.EVAL_PERIOD 10 \
  TRAIN.CHECKPOINT_PERIOD 25 \
  SOLVER.EPOCH_CYCLE 5.0 \
  SOLVER.BASE_LR 0.001 \
  SOLVER.LR_POLICY cosine \
  SOLVER.COSINE_END_LR 0.00001 \
  SOLVER.WEIGHT_DECAY 0.01 \
  SOLVER.OPTIMIZING_METHOD adam \
  SOLVER.MAX_EPOCH 100
  """
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from train_net import train, test_implementation
from train_net_des import train_des, test_implementation_des

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
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
