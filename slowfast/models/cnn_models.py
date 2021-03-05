import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class CNN_MLP(nn.Module):
    """
    Implemetation of a baseline CNN+MLP model for Clevrer
    """

    def init_params(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.fill_(0.0)
        elif type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)

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
        #self.frame_enc_dim = 512
        self.frame_enc_dim = 32
        self.cnn = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=self.frame_enc_dim)
        #Question Embedding
        #self.question_enc_dim = 128
        self.question_enc_dim = 16
        self.embed_layer = nn.Embedding(self.vocab_len, self.question_enc_dim, padding_idx=1) #Index 1 is for pad token
        
        #Prediction head MLP
        hid_dim = 2048
        hid_dim_2 = 2048
        hid_dim_3 = 1024
        self.pre_pred_head = nn.Sequential(
            nn.Linear(self.question_enc_dim + self.frame_enc_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hid_dim, hid_dim_2),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        #Question especific
        self.des_pred_head = nn.Sequential(
            nn.Linear(hid_dim_2, hid_dim_3),
            nn.ReLU(),
            nn.Linear(hid_dim_3, self.ans_vocab_len)
        )
        #Multiple choice answer => outputs a vector of size 4, 
        # which is interpreted as 4 logits, one for each binary classification of each choice
        self.mc_pred_head = nn.Sequential(
            nn.Linear(hid_dim_2, hid_dim_3),
            nn.ReLU(),
            nn.Linear(hid_dim_3, 4)
        )

        #Init parameters
        self.cnn.apply(self.init_params)
        self.pre_pred_head.apply(self.init_params)
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
        #Receives a batch of frames. To apply a CNN we can join the batch and time dimensions
        cb_sz = clips_b.size()
        frame_encs = self.cnn(clips_b.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
        frame_encs = frame_encs.view(cb_sz[0], cb_sz[1], self.frame_enc_dim) #Returns to batch format
        frame_encs = torch.sum(frame_encs, dim=1) / cb_sz[1] #Average frame encodings in a clip
        #Question embbeding and aggregation
        word_encs = self.embed_layer(question_b)
        q_len = word_encs.size()[1]
        word_encs = torch.sum(word_encs, dim=1) / q_len #Average word encodings in a question
        #Concatenate question and video encodings
        input_encs = torch.cat((frame_encs, word_encs), dim=1)
        #MLP
        input_encs = self.pre_pred_head(input_encs)
        if is_des_q:
            return self.des_pred_head(input_encs)
        else:
            return self.mc_pred_head(input_encs)




#__--____--____---___-LSTM__--____--____---___-

@MODEL_REGISTRY.register()
class CNN_LSTM(nn.Module):
    """
    Implemetation of a baseline CNN+LSTM model for Clevrer
    First receives the sequence of word embeddings for the question, 
    then the CNN embbedings for the frames
    """

    def init_params(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.fill_(0.0)
        elif type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        
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
        print("CNN_LSTM model")
        super(CNN_LSTM, self).__init__()
        #CUDA
        self.num_gpus = cfg.NUM_GPUS
        #Dataset specific parameters
        self.vocab_len = vocab_len
        self.ans_vocab_len = ans_vocab_len
        self.vocab = vocab
        #Input dimension for LSTM
        self.enc_dim = cfg.WORD_EMB.EMB_DIM
        #ResNet
        self.frame_enc_dim = self.enc_dim
        self.cnn = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=self.frame_enc_dim)

        #Question Embedding
        self.question_enc_dim = self.enc_dim
        self.embed_layer = nn.Embedding(self.vocab_len, self.question_enc_dim, padding_idx=1) #Index 1 is for pad token
        if cfg.WORD_EMB.USE_PRETRAINED_EMB:
            weights_matrix = self.parse_glove_file(cfg.WORD_EMB.GLOVE_PATH, self.enc_dim, self.vocab)
            self.embed_layer.load_state_dict({'weight': weights_matrix})
        if not cfg.WORD_EMB.TRAINABLE:
            self.embed_layer.weight.requires_grad = False

        #LSTM
        self.hid_st_dim = cfg.CLEVRERMAIN.LSTM_HID_DIM
        self.num_layers = 2
        self.num_directions = 2
        self.LSTM = torch.nn.LSTM(
            input_size=self.enc_dim, hidden_size=self.hid_st_dim, num_layers=self.num_layers,
            bias=True, batch_first=True, dropout=cfg.CLEVRERMAIN.T_DROPOUT, bidirectional=True
        )
        #Prediction head MLP
        hid_dim = 2048
        hid_dim_2 = 2048
        ph_input_dim = self.hid_st_dim*2
        #Question especific
        self.des_pred_head = nn.Sequential(
            nn.Linear(ph_input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.CLEVRERMAIN.T_DROPOUT),
            nn.Linear(hid_dim, hid_dim_2),
            nn.ReLU(),
            nn.Dropout(p=cfg.CLEVRERMAIN.T_DROPOUT),
            nn.Linear(hid_dim_2, self.ans_vocab_len)
        )
        #Multiple choice answer => outputs a vector of size 4, 
        # which is interpreted as 4 logits, one for each binary classification of each choice
        self.mc_pred_head = nn.Sequential(
            nn.Linear(ph_input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.CLEVRERMAIN.T_DROPOUT),
            nn.Linear(hid_dim, hid_dim_2),
            nn.ReLU(),
            nn.Dropout(p=cfg.CLEVRERMAIN.T_DROPOUT),
            nn.Linear(hid_dim_2, 4)
        )

        #Init parameters
        self.cnn.apply(self.init_params)
        #self.LSTM.apply(self.init_params)
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
        #Receives a batch of frames. To apply a CNN we can join the batch and time dimensions
        cb_sz = clips_b.size()
        frame_encs = self.cnn(clips_b.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
        frame_encs = frame_encs.view(cb_sz[0], cb_sz[1], self.frame_enc_dim) #Returns to batch format
        #Question embbeding and aggregation
        word_encs = self.embed_layer(question_b)
        #Concatenate question and video encodings
        rnn_input = torch.cat((word_encs, frame_encs), dim=1)
        #LSTM
        _, (h_n, _) = self.LSTM(rnn_input)
        x = torch.cat((h_n[-1], h_n[-2]), dim=1) #Cat forward and backward
        if is_des_q:
            return self.des_pred_head(x)
        else:
            return self.mc_pred_head(x)


