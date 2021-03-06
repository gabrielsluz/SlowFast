import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from .build import MODEL_REGISTRY

#__--____--____---___-LSTM__--____--____---___-

@MODEL_REGISTRY.register()
class TEXT_LSTM(nn.Module):
    """
    Implemetation of a baseline LSTM model for Clevrer
    Only uses the question
    Uses pretrained embeddings
    """

    def init_params(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.fill_(0.0)

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
        print("TEXT_LSTM model")
        super(TEXT_LSTM, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.vocab = vocab
        #Input dimension for LSTM
        self.enc_dim = cfg.WORD_EMB.EMB_DIM
        #Question Embedding
        self.embed_layer = nn.Embedding(self.vocab_len, self.enc_dim, padding_idx=1) #Index 1 is for pad token
        if cfg.WORD_EMB.USE_PRETRAINED_EMB:
            weights_matrix = self.parse_glove_file(cfg.WORD_EMB.GLOVE_PATH, self.enc_dim, self.vocab)
            self.embed_layer.load_state_dict({'weight': weights_matrix})
        if not cfg.WORD_EMB.TRAINABLE:
            self.embed_layer.weight.requires_grad = False
            
        #LSTM
        self.hid_st_dim = 256
        self.num_layers = 2
        self.num_directions = 2 #Check bellow: parameter bidirectional
        self.LSTM = torch.nn.LSTM(
            input_size=self.enc_dim, hidden_size=self.hid_st_dim, num_layers=self.num_layers,
            bias=True, batch_first=True, dropout=0.5, bidirectional=(self.num_directions == 2)
        )

        # self.des_pred_head = nn.Linear(self.hid_st_dim*2, self.ans_vocab_len)
        # self.mc_pred_head = nn.Linear(self.hid_st_dim*2, 4)

        #Prediction head MLP
        hid_dim = 2048
        hid_dim_2 = 1024
        input_dim = self.hid_st_dim*2
        dropout_p = 0.5
        #Question especific
        self.des_pred_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid_dim, hid_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid_dim_2, self.ans_vocab_len)
        )
        #Multiple choice answer => outputs a vector of size 4, 
        # which is interpreted as 4 logits, one for each binary classification of each choice
        self.mc_pred_head = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid_dim, hid_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hid_dim_2, 4)
        )

        #Init parameters
        self.des_pred_head.apply(self.init_params)
        self.mc_pred_head.apply(self.init_params)

    def forward(self, question_b, is_des_q):
        """
        Receives a batch of clips and questions:
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
                is_des_q (bool): Indicates if is descriptive question or multiple choice
        """
        #Question embbeding and aggregation
        embs = self.embed_layer(question_b)
        #LSTM
        _, (h_n, _) = self.LSTM(embs)
        x = torch.cat((h_n[-1], h_n[-2]), dim=1) #Cat forward and backward
        if is_des_q:
            return self.des_pred_head(x)
        else:
            return self.mc_pred_head(x)


#__--____--____---___-GRU__--____--____---___-

@MODEL_REGISTRY.register()
class TEXT_GRU(nn.Module):
    """
    Implemetation of a baseline GRU model for Clevrer
    Only uses the question
    Uses pretrained embeddings
    """

    def init_params(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.fill_(0.0)

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
        print("TEXT_GRU model")
        super(TEXT_GRU, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.vocab = vocab
        #Input dimension for LSTM
        self.enc_dim = cfg.WORD_EMB.EMB_DIM
        #Question Embedding
        self.embed_layer = nn.Embedding(self.vocab_len, self.enc_dim, padding_idx=1) #Index 1 is for pad token
        if cfg.WORD_EMB.USE_PRETRAINED_EMB:
            weights_matrix = self.parse_glove_file(cfg.WORD_EMB.GLOVE_PATH, self.enc_dim, self.vocab)
            self.embed_layer.load_state_dict({'weight': weights_matrix})
        if not cfg.WORD_EMB.TRAINABLE:
            self.embed_layer.weight.requires_grad = False
            
        #GRU
        self.hid_st_dim = 256
        self.num_layers = 2
        self.num_directions = 2 #Check bellow: parameter bidirectional
        self.GRU = torch.nn.GRU(
            input_size=self.enc_dim, hidden_size=self.hid_st_dim, num_layers=self.num_layers,
            bias=True, batch_first=True, dropout=0.0, bidirectional=True
        )
        
        self.des_pred_head = nn.Linear(self.hid_st_dim*2, self.ans_vocab_len)
        self.mc_pred_head = nn.Linear(self.hid_st_dim*2, 4)

        #Prediction head MLP
        # hid_dim = 2048
        # hid_dim_2 = 1024
        # input_dim = self.hid_st_dim*2
        # dropout_p = 0.0
        # #Question especific
        # self.des_pred_head = nn.Sequential(
        #     nn.Linear(input_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(hid_dim, hid_dim_2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(hid_dim_2, self.ans_vocab_len)
        # )
        # #Multiple choice answer => outputs a vector of size 4, 
        # # which is interpreted as 4 logits, one for each binary classification of each choice
        # self.mc_pred_head = nn.Sequential(
        #     nn.Linear(input_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(hid_dim, hid_dim_2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(hid_dim_2, 4)
        # )

        #Init parameters
        self.des_pred_head.apply(self.init_params)
        self.mc_pred_head.apply(self.init_params)

    def forward(self, question_b, is_des_q):
        """
        Receives a batch of clips and questions:
                question_b (tensor): The dimension is
                    `batch_size` x 'max sequence length'
                is_des_q (bool): Indicates if is descriptive question or multiple choice
        """
        #Question embbeding and aggregation
        embs = self.embed_layer(question_b)
        #GRU
        _, h_n = self.GRU(embs)
        x = torch.cat((h_n[-1], h_n[-2]), dim=1) #Cat forward and backward
        if is_des_q:
            return self.des_pred_head(x)
        else:
            return self.mc_pred_head(x)
