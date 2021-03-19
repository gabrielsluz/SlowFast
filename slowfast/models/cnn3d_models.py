import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from slowfast.models.video_model_builder import SlowFast

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class CNN_3D_LSTM(nn.Module):
    """
    Implemetation of a baseline SlowFast+LSTM model for Clevrer
    First receives the sequence of word embeddings for the question, 
    then the CNN embbedings for the frames
    """

    def init_params(self, layer):
        if type(layer) == nn.Embedding:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.weight[layer.padding_idx])
        elif type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(layer.bias)
        # elif type(layer) == nn.LSTM:
        #     for param in layer.parameters():
        #         if len(param.shape) >= 2:
        #             nn.init.orthogonal_(param.data)
        #             # nn.init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
        #         else:
        #             nn.init.normal_(param.data)
        # elif type(layer) == nn.LSTMCell:
        #     for param in layer.parameters():
        #         if len(param.shape) >= 2:
        #             nn.init.orthogonal_(param.data)
        #             # nn.init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
        #         else:
        #             nn.init.normal_(param.data)

    def parse_glove_file(self, file_name, emb_dim, vocab_dict):
        """
        Opens a Glove pretrained embeddings file with embeddings with dimension emb_dim
        Builds a matrix vocab_size x emb_dim, compatible with nn.Embedding to be used with vocab_dict
        """
        word_list = []
        for word in vocab_dict.keys():
            word_list.append(word)
        emb_mat = np.zeros((len(vocab_dict), emb_dim))
        with open(file_name, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if not word in vocab_dict:
                    continue
                vect = np.array(line[1:]).astype(np.float)
                emb_mat[vocab_dict[word]] = vect
                word_list.remove(word)

        if len(word_list) > 0:
            print("Missing following words in pretrained embeddings")
            print(word_list)
        return torch.from_numpy(emb_mat)

    def __init__(self, cfg, vocab_len, ans_vocab_len, vocab):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        print("CNN_3D_LSTM model")
        super(CNN_3D_LSTM, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.vocab = vocab
        #SlowFast
        self.frame_enc_dim = cfg.MODEL.NUM_CLASSES
        self.SlowFast = SlowFast(cfg)
        #Question Embedding
        #Input dimension for LSTM
        self.question_enc_dim = cfg.WORD_EMB.EMB_DIM
        self.embed_layer = nn.Embedding(self.vocab_len, self.question_enc_dim, padding_idx=1) #Index 1 is for pad token
        if cfg.WORD_EMB.USE_PRETRAINED_EMB:
            weights_matrix = self.parse_glove_file(cfg.WORD_EMB.GLOVE_PATH, self.question_enc_dim, self.vocab)
            self.embed_layer.load_state_dict({'weight': weights_matrix})
        else:
            self.embed_layer.apply(self.init_params)
        if not cfg.WORD_EMB.TRAINABLE:
            self.embed_layer.weight.requires_grad = False

        #LSTM
        self.hid_st_dim = cfg.CLEVRERMAIN.LSTM_HID_DIM
        self.num_layers = 2
        self.num_directions = 2
        self.LSTM = torch.nn.LSTM(
            input_size=self.question_enc_dim, hidden_size=self.hid_st_dim, num_layers=self.num_layers,
            bias=True, batch_first=True, dropout=cfg.CLEVRERMAIN.T_DROPOUT, bidirectional=True
        )
        #Prediction head MLP
        hid_dim = 2048
        ph_input_dim = self.hid_st_dim + self.frame_enc_dim
        #Question especific
        self.des_pred_head = nn.Sequential(
            nn.Linear(ph_input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, self.ans_vocab_len)
        )
        #Multiple choice answer => outputs a vector of size 4, 
        # which is interpreted as 4 logits, one for each binary classification of each choice
        self.mc_pred_head = nn.Sequential(
            nn.Linear(ph_input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 4)
        )

        #Init parameters *embed layer is initialized above
        self.LSTM.apply(self.init_params)
        self.des_pred_head.apply(self.init_params)
        self.mc_pred_head.apply(self.init_params)

    def forward(self, clips_b, question_b, is_des_q):
        """
        Receives a batch of clips and questions:
                clips_b (tensor): the frames of sampled from the video. The dimension
                    is `batch_size` x `num frames` x `channel` x `height` x `width`.
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
                is_des_q (bool): Indicates if is descriptive question or multiple choice
        """
        #Receives a batch of frames
        cb_sz = clips_b.size()
        frame_encs = self.SlowFast(clips_b)
        #Question embbeding
        word_encs = self.embed_layer(question_b)
        #LSTM
        _, (h_n, _) = self.LSTM(word_encs)
        words_x = torch.cat((h_n[-1], h_n[-2]), dim=1) #Cat forward and backward
        x = torch.cat((words_x, frame_encs), dim=1)
        if is_des_q:
            return self.des_pred_head(x)
        else:
            return self.mc_pred_head(x)