import torch
import torch.nn as nn
import torch.nn.functional as F

from .monet import Monet
from .transformer import Transformer

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
        #Uses state_dict
        self.Monet.load_state_dict(torch.load(cfg.MONET.STATE_DICT_PATH))

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

    
    def assemble_input(self):
        """
        Assembles the input sequence for the Transformer.
        Receives: slots, word embeddings
        Sequence: <CLS, slots, words>
        The slots and words are concatenated with a one hot that indicates
        if they are slots or words. => Sequence vectors are d + 2 dimensional
        """
        pass

    def forward(self, ):
        src = self.embed_layer(src) * math.sqrt(self.input_dim)