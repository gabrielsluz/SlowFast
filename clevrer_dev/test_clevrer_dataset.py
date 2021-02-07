#!/usr/bin/env python3
from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

frame_dataset = Clevrer(cfg, 'train')
print(len(frame_dataset))
if len(frame_dataset) > 5:
    max_i = 5
else:
    max_i = len(frame_dataset)
for i in range(max_i):
    print(frame_dataset[i].size())

tensor_image = frame_dataset[0].permute(1,2,0)
plt.imshow(tensor_image)
plt.savefig('sample_frame.png')