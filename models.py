""" 
    Custom Crytpo-Prediction RNN models definition.
    ---------
    Notations:
        B=batch_size, D=features_dim, L=sentence_length, H=hidden_dim, N=n_layers, M=n_directions

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # softmax     

AVAILABLE_CELLS = ["RNN", "LSTM", "GRU"] # cell types

class Encoder(nn.Module):
    """ Encoder Class for CryptoRegressor. """
    def __init__(self, features_dim, hidden_dim, n_layers, dropout, bidirectional, cell_type):
        assert cell_type in AVAILABLE_CELLS, f"Invalid parameter, cell_type must be one of {AVAILABLE_CELLS}."

        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        cell = getattr(torch.nn, cell_type) # fetch constructor from torch.nn
        self.rnn = cell(features_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs):
        outputs, hidden = self.rnn(inputs)
        return outputs, hidden

class Attention(nn.Module):
    """ ScaledDotProduct attention, as in 'Attention is all you need'. """
    def __init__(self, query_dim, key_dim, value_dim):        
        super(Attention, self).__init__()
        self.query_dim = query_dim

    def forward(self, query, keys, values):
        # Query=[BxQ]   Keys=[BxKxT]   Values=[BxTxV]
        # returns = a:[HxB], lin_comb:[BxV]
        
        query = query.unsqueeze(1) # [BxQ] --> [Bx1xQ]
        score = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] --> [Bx1xT], assuming Q=K
        score = score.mul(1./np.sqrt(self.query_dim)) # scale
        attention = F.softmax(score, dim=2) # normalize
        context = torch.bmm(attention, values).squeeze(1) # [Bx1xT].[BxTxV] -> [BxV]
        return context, attention

class CryptoRegressor(nn.Module):
    """ CryptoRegressor. """
    def __init__(self, encoder, attention, hidden_dim, output_dim, dropout=0):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.output_dim = output_dim
        self.regressor = nn.Linear(hidden_dim, output_dim) # regressor
        self.dropout = nn.Dropout(dropout)

        self.use_attention = (attention is not None)

        # stats
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)   
        print(f"Model has a total of {n_params:,} trainable parameters.")     
    
    def forward(self, inputs):
        outputs, hidden = self.encoder(inputs) # outputs=[L x B x H*M]
        if isinstance(hidden, tuple): # LSTM
            hidden, cell = hidden # hidden=[N*M x B x H]

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1) # [BxH]
        else:
            hidden = hidden[-1]

        if self.use_attention:
            query = hidden
            keys = outputs.transpose(0,1).transpose(1,2) # [N*MxBxH] -> [BxHxN*M]
            values = outputs.transpose(0,1) # [N*MxBxH] -> [BxN*MxH]
            context, attention = self.attention(query, keys, values)
            context = self.dropout(context)
            logits = self.regressor(context)
            return logits

        hidden = self.dropout(hidden) # apply dropout before regressor
        logits = self.regressor(hidden)
        return logits