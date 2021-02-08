from slowfast.models.monet import Monet
from collections import namedtuple
import slowfast.utils.checkpoint as cu

from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

import numpy as np

args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

frame_dataset = Clevrer(cfg, 'train')
print(len(frame_dataset))

config_options = [
    # Training config
    #'vis_every',  # Visualize progress every X iterations
    #'batch_size',
    #'num_epochs',
    #'load_parameters',  # Load parameters from checkpoint
    #'checkpoint_file',  # File for loading/storing checkpoints
    #'data_dir',  # Directory for the training data
    #'parallel',  # Train using nn.DataParallel
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
]

MonetConfig = namedtuple('MonetConfig', config_options)

clevr_conf = MonetConfig(num_slots=8,
                        num_blocks=6,
                        channel_base=64,
                        bg_sigma=0.09,
                        fg_sigma=0.11,
                        )
monet_model = Monet(clevr_conf, 64, 64)
cu.load_checkpoint("/datasets/checkpoint_epoch_00020.pyth", monet_model,data_parallel=False)

def visualize_masks(imgs, masks, recons):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    seg_maps = seg_maps[0].transpose((1, 2, 0))
    plt.imshow(seg_maps)
    plt.savefig('seg_maps.png')

print(frame_dataset[0][0].size())
tensor_input = torch.unsqueeze(frame_dataset[0][0][5], 0)
output = monet_model.forward(tensor_input)
print(output['reconstructions'].size())

tensor_image = output['reconstructions'][0].permute(1,2,0).detach().numpy()
# plt.imshow(tensor_image)
# plt.savefig('sample_frame.png')
imgs = tensor_input
masks = output['masks'].detach().numpy()
recons = output['reconstructions'].detach().numpy()
visualize_masks(imgs, masks, recons)
tensor_image = tensor_input[0].permute(1,2,0)
plt.imshow(tensor_image)
plt.savefig('sample_frame.png')

"""
python3 clevrer_dev/visualize_monet.py \
  --cfg clevrer_dev/clevrer_frame.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  NUM_GPUS 0 \
  TRAIN.BATCH_SIZE 3 \
  SOLVER.MAX_EPOCH 1
"""