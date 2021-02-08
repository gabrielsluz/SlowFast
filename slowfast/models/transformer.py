"""
Made on top of Pytorch official tutorial:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayerPositional(TransformerEncoderLayer):
    """
    Extends the TransformerEncoderLayer to apply Positional Encoding
    at each layer
    """
    def __init__(self, input_dim, nhead, hid_dim=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayerPositional, self).__init__(input_dim, nhead, hid_dim, dropout, activation)
        self.pos_emb = PositionalEmbedding(input_dim, dropout)
    
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src = self.pos_emb(src)
        return super(TransformerEncoderLayerPositional, self).forward(src, src_mask, src_key_padding_mask)



class TransformerModel(nn.Module):
    """"
    Implementation of the Transformer Model used in the Main model.
    As BERT, it only uses the Encoder part of the Transformer.
    It receives a sequence of vectors and outputs a sequence of transformed vectors
    It uses Positional Encoding at each layer
    """

    def __init__(self, vocab_len, input_dim, nhead, hid_dim, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, hid_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_len, input_dim)
        self.input_dim = input_dim
        #self.decoder = nn.Linear(input_dim, vocab_len) #Wrong, in our model will be very different
        #It will be an MLP, but out of here

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def assemble_input(self): #Will be taken out of here
        """
        Assembles the input sequence for the Transformer.
        Receives: slots, word embeddings
        Sequence: <CLS, slots, words>
        The slots and words are concatenated with a one hot that indicates
        if they are slots or words. => Sequence vectors are d + 2 dimensional
        """
        pass

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output