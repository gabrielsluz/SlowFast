#!/usr/bin/env python3
from slowfast.datasets.clevrer_text import Clevrertext, Clevrertext_des, Clevrertext_mc
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging
from slowfast.models.build import MODEL_REGISTRY
import slowfast.models.losses as losses

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/text_baseline/test_clevrer_text.py \
  --cfg clevrer_dev/text_baseline/text_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  WORD_EMB.USE_PRETRAINED_EMB True \
  WORD_EMB.TRAINABLE True \
  DATA.PATH_PREFIX /datasets/clevrer_dummy

python3 clevrer_dev/text_baseline/test_clevrer_text.py \
  --cfg clevrer_dev/text_baseline/text_gru.yaml \
  DATA.PATH_TO_DATA_DIR /content/clevrer \
  WORD_EMB.USE_PRETRAINED_EMB True \
  WORD_EMB.TRAINABLE True \
  DATA.PATH_PREFIX /content/clevrer
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)

dataset = Clevrertext(cfg, 'val')
dataset_des = Clevrertext_des(cfg, 'val')
dataset_mc = Clevrertext_mc(cfg, 'val')
print("Dataset len = {}".format(len(dataset)))
print("Dataset Des len = {}".format(len(dataset_des)))
print("Dataset Mc len = {}".format(len(dataset_mc)))

if len(dataset) > 3:
    max_i = 3
else:
    max_i = len(dataset)
print("First {} items".format(max_i))
for i in range(max_i):
    print(dataset_des.get_video_info(i))
    print(dataset_mc.get_video_info(i))

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0)
print("Dataloader:")
for i_batch, sample_batched in enumerate(dataloader):
    print("is_des = {}".format(sample_batched['question_dict']['is_des']))
    print("Question = {}".format(sample_batched['question_dict']['question']))
    print("Ans = {}".format(sample_batched['question_dict']['ans']))
    print("Len = {}".format(sample_batched['question_dict']['len']))
    print("Index = {}".format(sample_batched['index'].size()))

    print(dataset.get_video_info(sample_batched['index'][0]))
    is_des = sample_batched['question_dict']['is_des']
    question = sample_batched['question_dict']['question']
    ans = sample_batched['question_dict']['ans']
    break

print("Model")
vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()
vocab = dataset.get_vocab()
name = cfg.MODEL.MODEL_NAME
model = MODEL_REGISTRY.get(name)(cfg, vocab_len, ans_vocab_len, vocab)

print("Embedding layer: ")
print(model.embed_layer.weight)

print("Pass through model")
print("Question = {}".format(question))
if is_des:
    pred_des_ans = model(question, True)
    print("Model output = {}".format(pred_des_ans))
    des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
    loss = des_loss_fun(pred_des_ans, ans)
else:
    pred_mc_ans = model(question, False)
    print("Model output = {}".format(pred_mc_ans))
    mc_loss_fun = losses.get_loss_func('bce_logit')(reduction="mean")
    loss = mc_loss_fun(pred_mc_ans, ans)

print("Loss = {}".format(loss))
loss.backward()
print("Embed Grad 0:5:")
print(model.embed_layer.weight.grad[0:5])