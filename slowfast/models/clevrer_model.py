import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .monet import Monet
from .transformer import Transformer
from .build import MODEL_REGISTRY

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

@MODEL_REGISTRY.register()
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
        super(ClevrerMain, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.slot_dim = 16
        self.trans_dim = self.slot_dim + 2 #Dimension of the transformer inputs

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
        #Is MONet not trainable ?
        if not cfg.CLEVRERMAIN.MONET_TRAINABLE:
            for param in self.Monet.parameters():
                param.requires_grad = False

        #Embedding for the words
        self.embed_layer = nn.Embedding(self.vocab_len, self.slot_dim)

        #Transformer setup
        self.Transformer = Transformer(input_dim=self.trans_dim, 
                                        nhead=cfg.CLEVRERMAIN.T_HEADS, hid_dim=cfg.CLEVRERMAIN.T_HID_DIM, 
                                        nlayers=cfg.CLEVRERMAIN.T_LAYERS, dropout=cfg.CLEVRERMAIN.T_DROPOUT)

        #Prediction head MLP
        #TODO: Currently only for descriptive questions
        self.pred_head = nn.Sequential(
            nn.Linear(self.trans_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, self.ans_vocab_len)
        )

    
    def assemble_input(self, slots_b, word_embs_b):
        """
        Assembles the input sequence for the Transformer.
        Receives: slots, word embeddings
            slots_b(tensor): Batch x T*Num_slots x Slot_dim
            word_embs_b(tensor): Batch x Question_len x Slot_dim
        Returns: Sequence: <CLS, slots, words>
            (tensor): Batch x (1 + T*Num_slots + Question_len) x (Slot_dim + 2)
        The slots and words are concatenated with a one hot that indicates
        if they are slots or words. => Sequence vectors are d + 2 dimensional
        """
        #CLS is token 0
        #Test if tensor is in cuda: next(model.parameters()).is_cuda
        #Tensor setups:
        batch_size = slots_b.size()[0]

        cls_indexes = torch.zeros((batch_size, 1), dtype=torch.long)
        z_cls = torch.zeros((batch_size, 1, 2))
        o_slots = torch.ones((batch_size, slots_b.size()[1], 1))
        z_slots = torch.zeros((batch_size, slots_b.size()[1], 1))
        o_words = torch.ones((batch_size, word_embs_b.size()[1], 1))
        z_words = torch.zeros((batch_size, word_embs_b.size()[1], 1))
        if self.num_gpus:
            cur_device = torch.cuda.current_device()
            cls_indexes = cls_indexes.cuda(device=cur_device)
            z_cls = z_cls.cuda(device=cur_device)
            o_slots = o_slots.cuda(device=cur_device)
            z_slots = z_slots.cuda(device=cur_device)
            o_words = o_words.cuda(device=cur_device)
            z_words = z_words.cuda(device=cur_device)

        #CLS
        cls_t = self.embed_layer(cls_indexes)
        cls_t = torch.cat((cls_t, z_cls), dim=2)
        #Slots
        slots_b = torch.cat((slots_b, o_slots, z_slots), dim=2)
        #Words
        word_embs_b = torch.cat((word_embs_b, z_words, o_words), dim=2)
        return torch.cat((cls_t, slots_b, word_embs_b), dim=1)

    def forward(self, clips_b, question_b):
        """
        Receives a batch of clips and questions:
                clips_b (tensor): the frames of sampled from the video. The dimension
                    is `batch_size` x `num frames` x `channel` x `height` x `width`.
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        word_embs_b = self.embed_layer(question_b) * math.sqrt(self.slot_dim)
        end.record()
        torch.cuda.synchronize()
        print("Embedding layer time = {}".format(start.elapsed_time(end)))
        batch_size = clips_b.size()[0]
        slots_l = []

        start.record()
        for i in range(batch_size):
            slots_l.append(self.Monet.return_means(clips_b[i]))
        end.record()
        torch.cuda.synchronize()
        print("MONet time = {}".format(start.elapsed_time(end)))
        slots_b = torch.stack(slots_l, dim=0)

        start.record()
        transformer_in = self.assemble_input(slots_b, word_embs_b)
        end.record()
        torch.cuda.synchronize()
        print("Assemble input time = {}".format(start.elapsed_time(end)))
        start.record()
        transformer_out = self.Transformer(transformer_in)
        end.record()
        torch.cuda.synchronize()
        print("Transformer time = {}".format(start.elapsed_time(end)))
        start.record()
        desc_ans = self.pred_head(transformer_out[:, 0])
        end.record()
        torch.cuda.synchronize()
        print("Pred head time = {}".format(start.elapsed_time(end)))
        return desc_ans

        