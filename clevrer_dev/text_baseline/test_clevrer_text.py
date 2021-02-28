#!/usr/bin/env python3
from slowfast.datasets.clevrer_text import Clevrertext
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging
from slowfast.models.build import MODEL_REGISTRY

"""
Prints information about the dataset for testing and debugging

Example:
python3 clevrer_dev/text_baseline/test_clevrer_text.py \
  --cfg clevrer_dev/text_baseline/text_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy
"""

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

dataset = Clevrertext(cfg, 'train')
print("Dataset len = {}".format(len(dataset)))

if len(dataset) > 5:
    max_i = 5
else:
    max_i = len(dataset)
print("First {} items".format(max_i))
for i in range(max_i):
    print(dataset.get_video_info(i))

# tensor_image = dataset[0][0][0].permute(1,2,0)
# plt.imshow(tensor_image)
# plt.savefig('sample_frame.png')

#Test DataLoader
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0)
print("Dataloader:")
for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['question_dict']['des_q'].size())
    print(sample_batched['question_dict']['des_ans'].size())
    print(sample_batched['question_dict']['mc_q'].size())
    print(sample_batched['question_dict']['mc_ans'].size())
    print(sample_batched['index'].size())

    print(dataset.get_video_info(sample_batched['index'][0]))
    example_batch = sample_batched
    break

print("Model")
vocab_len = dataset.get_vocab_len()
ans_vocab_len = dataset.get_ans_vocab_len()
vocab = dataset.get_vocab()
name = cfg.MODEL.MODEL_NAME
model = MODEL_REGISTRY.get(name)(cfg, vocab_len, ans_vocab_len, vocab)

print("Embedding layer: ")
print(model.embed_layer.weight)