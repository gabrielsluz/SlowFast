import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class CNN_MLP(nn.Module):
    """
    Implemetation of a baseline CNN+MLP model for Clevrer
    Uses pretrained word embeddings
    """
    def __init__(self, cfg, vocab_len, ans_vocab_len):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(CNN_MLP, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        #ResNet
        frame_enc_dim = 512
        self.cnn = torchvision.models.resnet18(pretrained=False, progress=True, num_classes= frame_enc_dim)
        #Question Embedding
        question_enc_dim = 128
        self.embed_layer = nn.Embedding(self.vocab_len, question_enc_dim)
        # #Prediction head MLP
        # self.des_pred_head = nn.Sequential(
        #     nn.Linear(self.trans_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
        #     nn.ReLU(),
        #     nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, self.ans_vocab_len)
        # )
        # #Multiple choice answer => outputs a vector of size 4, 
        # # which is interpreted as 4 logits, one for each binary classification of each choice
        # self.mc_pred_head = nn.Sequential(
        #     nn.Linear(self.trans_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
        #     nn.ReLU(),
        #     nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, 4)
        # )

    def forward(self, clips_b, question_b, is_des_q):
        """
        Receives a batch of clips and questions:
                clips_b (tensor): the frames of sampled from the video. The dimension
                    is `batch_size` x `num frames` x `channel` x `height` x `width`.
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
                is_des_q (bool): Indicates if is descriptive question or multiple choice
        """
        x = clips_b[0]
        x = self.cnn(x) #Checar se consigo aplicar resnet em um batch de videos. Nao diretamente.
        #Agregar embeddings
        #Concatenar e passar pela mlp
        return x

        