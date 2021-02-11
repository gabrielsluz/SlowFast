import torch
import torch.nn as nn
import torch.nn.functional as F

from .monet import Monet
from .transformer import Transformer

import slowfast.utils.checkpoint as cu

from collections import namedtuple

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

class ClevrerMain(nn.Module):
    """
    Implemetation of the main Model for the Clevrer Dataset
    It combines the Transformer and MONet models

    *The original paper of MONet and of this algorithm use a latent represantion
    of 16 dimensions. This number is hardcoded in the MONet model.
    TODO:
        - Multiple Choice questions
        - Self Supervision
        - LAMB optimizer
    """
    def __init__(self, cfg, vocab_len, ans_vocab_len):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.slot_dim = 16

        #MONet setup
        clevr_conf = MonetConfig(num_slots=cfg.MONET.NUM_SLOTS,
                           num_blocks=6,
                           channel_base=64,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                          )
        self.Monet = Monet(clevr_conf, cfg.DATA.RESIZE_H, cfg.DATA.RESIZE_W)
        #MONet should have been pretrained => load it at this step 
        cu.load_checkpoint(cfg.MONET.CHECKPOINT_LOAD, self.Monet, data_parallel=False)

        #Embedding for the words
        self.embed_layer = nn.Embedding(self.vocab_len, self.slot_dim)

        #Transformer setup
        self.Transformer = Transformer(input_dim=self.slot_dim, 
                                        nhead=cfg.CLEVRERMAIN.T_HEADS, hid_dim=cfg.CLEVRERMAIN.T_HID_DIM, 
                                        nlayers=CLEVRERMAIN.T_LAYERS, dropout=cfg.CLEVRERMAIN.T_DROPOUT)

        #Prediction head MLP
        #TODO: Currently only for descriptive questions
        self.pred_head = nn.Sequential(
            nn.Linear(self.slot_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, self.ans_vocab_len)
        )

    
    def assemble_input(self, slots_b, word_embs_b):
        """
        Assembles the input sequence for the Transformer.
        Receives: slots, word embeddings
        Sequence: <CLS, slots, words>
        The slots and words are concatenated with a one hot that indicates
        if they are slots or words. => Sequence vectors are d + 2 dimensional
        """
        #CLS is token 0
        #Test if tensor is in cuda: next(model.parameters()).is_cuda
        batch_size = slots_b.size()[0]
        cls_t = self.embed_layer(torch.zeros((batch_size + 2, 1), dtype=torch.long))
        o = torch.ones((batch_size,1))
        z = torch.zeros((batch_size,1))
        slots_b = torch.cat((slots_b, o, z), dim=1)
        word_embs_b = torch.cat((word_embs_b z, o), dim=1)
        return torch.cat(cls_t, slots_b, word_embs_b, dim=0)

    def forward(self, clips_b, question_b):
        """
        Receives a batch of clips and questions:
                clips_b (tensor): the frames of sampled from the video. The dimension
                    is `batch_size` x `num frames` x `channel` x `height` x `width`.
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
        """
        word_embs_b = self.embed_layer(question_b) * math.sqrt(self.slot_dim)

        batch_size = clips_b.size()[0]
        slots_l = []
        for i in range(batch_size):
            slots_l.append(self.Monet(clips_b[i]))
        slots_b = torch.stack(slots_l, dim=0)
        print(slots_b.size())
        print(word_embs_b.size())