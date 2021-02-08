from slowfast.models.monet import Monet
from collections import namedtuple
import slowfast.utils.checkpoint as cu

from slowfast.datasets.clevrer import Clevrer
import matplotlib.pyplot as plt
import torch

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

import numpy as np

"""
Code for visualizing the MONet reconstruction and masks
MONet's load checkpoint must be informed through cfg

Example:
python3 clevrer_dev/visualize_monet.py \
  --cfg clevrer_dev/visualize_monet.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer_dummy \
  DATA.PATH_PREFIX /datasets/clevrer_dummy \
  MONET.CHECKPOINT_LOAD /datasets/checkpoint_epoch_00020.pyth
"""

def visualize_masks(imgs, masks):
    '''
    Using colors, shows which pixel belongs to each mask
    '''
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
    return seg_maps

#Fetch dataset
args = parse_args()
cfg = load_config(args)

logger = logging.get_logger(__name__)
logging.setup_logging(cfg.OUTPUT_DIR)

frame_dataset = Clevrer(cfg, 'train')

#Load model
config_options = [
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
cu.load_checkpoint(cfg.MONET.CHECKPOINT_LOAD, monet_model, data_parallel=False)


#Sample input frame from a video and add another dimension
frame_idx = 15
tensor_input = torch.unsqueeze(frame_dataset[0][0][frame_idx], 0)
#tensor_input = frame_dataset[0][0] # To return an entire video
output = monet_model.forward(tensor_input)
print(output['reconstructions'].size())
print(output['masks'].size())
print(output['loss'])

#Save reconstruction
reconstruction_img = output['reconstructions'][0].permute(1,2,0).detach().numpy()
plt.imshow(reconstruction_img)
plt.savefig('reconstruction.png')

imgs = tensor_input
masks = output['masks'].detach().numpy()
recons = output['reconstructions'].detach().numpy()
seg_maps = visualize_masks(imgs, masks)
seg_maps = seg_maps[0].transpose((1, 2, 0))
plt.imshow(seg_maps)
plt.savefig('seg_maps.png')

tensor_image = tensor_input[0].permute(1,2,0)
plt.imshow(tensor_image)
plt.savefig('sample_frame.png')
