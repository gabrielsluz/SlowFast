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
        self.frame_enc_dim = 512
        self.cnn = torchvision.models.resnet18(pretrained=False, progress=True, num_classes= self.frame_enc_dim)
        #Question Embedding
        self.question_enc_dim = 128
        self.embed_layer = nn.Embedding(self.vocab_len, self.question_enc_dim)
        #Prediction head MLP
        self.des_pred_head = nn.Sequential(
            nn.Linear(self.question_enc_dim + self.frame_enc_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, self.ans_vocab_len)
        )
        #Multiple choice answer => outputs a vector of size 4, 
        # which is interpreted as 4 logits, one for each binary classification of each choice
        self.mc_pred_head = nn.Sequential(
            nn.Linear(self.question_enc_dim + self.frame_enc_dim, cfg.CLEVRERMAIN.PRED_HEAD_DIM),
            nn.ReLU(),
            nn.Linear(cfg.CLEVRERMAIN.PRED_HEAD_DIM, 4)
        )

    def forward(self, clips_b, question_b, is_des_q):
        """
        Receives a batch of clips and questions:
                clips_b (tensor): the frames of sampled from the video. The dimension
                    is `batch_size` x `num frames` x `channel` x `height` x `width`.
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
                is_des_q (bool): Indicates if is descriptive question or multiple choice
        """
        #Receives a batch of frames. To apply a CNN we can join the batch and time dimensions
        cb_sz = clips_b.size()
        frame_encs = self.cnn(clips_b.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
        frame_encs = frame_encs.view(cb_sz[0], cb_sz[1], self.frame_enc_dim) #Returns to batch format
        frame_encs = torch.sum(frame_encs, dim=1) / cb_sz[1] #Average frame encodings in a clip
        #Question embbeding and aggregation
        #May need to deal with padding
        word_encs = self.embed_layer(question_b)
        q_len = word_encs.size()[1]
        word_encs = torch.sum(word_encs, dim=1) / q_len #Average word encodings in a question
        #Concatenate question and video encodings
        input_encs = torch.cat((frame_encs, word_encs), dim=1)
        if is_des_q:
            return self.des_pred_head(input_encs)
        else:
            return self.mc_pred_head(input_encs)

        