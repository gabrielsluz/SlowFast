#!/usr/bin/env python3
from slowfast.datasets.clevrer_dual import Clevrer_des
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from slowfast.models.cnn_models import CNN_LSTM

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/tests/model_eval_test.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy
"""

args = parse_args()
cfg = load_config(args)
cfg.WORD_EMB.EMB_DIM = 5
cfg.CLEVRERMAIN.LSTM_HID_DIM = 5

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrer_des(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))
vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()
vocab = dataset.get_vocab()
model = CNN_LSTM(cfg, vocab_len, ans_vocab_len, vocab)


#Test DataLoader
dataloader = DataLoader(dataset, batch_size=2,
                        shuffle=False, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['frames'].size())
    print(sample_batched['question_dict']['question'].size())
    print(sample_batched['question_dict']['ans'].size())
    print(sample_batched['index'].size())
    frames = sample_batched['frames']
    des_q = sample_batched['question_dict']['question']
    des_ans = sample_batched['question_dict']['ans']
    pred_des_ans = model(frames, des_q, True)
    print("Pred_des_ans train = {}".format(pred_des_ans))

    model.eval()
    pred_des_ans = model(frames, des_q, True)
    print("Pred_des_ans val = {}".format(pred_des_ans))
    break

    