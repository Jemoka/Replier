#pylint: disable=E1101

import re
import csv
import math
import numpy as np

from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

#### Utilities ####
np2tens = lambda x:torch.from_numpy(x).float()

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

#### Network ####
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class Transformer(nn.Module):
    # def __init__(self, numberTokens:int, embeddingSize:int, attentionHead:int, hiddenDenseSize:int, numberLayers:int):
    def __init__(self, numberTokens, embeddingSize, numberTransformerLayers, attentionHeadCount, transformerHiddenDenseSize):
        # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.embeddingSize = embeddingSize

        self.embedding = nn.Embedding(numberTokens, embeddingSize)

        self.positional_encoding = self.positionalencoding1d(embeddingSize)
        
        encoderLayer = nn.TransformerEncoderLayer(embeddingSize, attentionHeadCount, transformerHiddenDenseSize)

        self.encoder = nn.TransformerEncoder(encoderLayer, numberTransformerLayers)

        self.decoder = nn.Linear(embeddingSize, numberTokens)

    @staticmethod
    def positionalencoding1d(d_model, length_max=5000):
        """
        PositionalEncoding2D: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        AttentionIsAllYouNeed: https://arxiv.org/pdf/1706.03762.pdf

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length_max, d_model)
        position = torch.arange(0, length_max).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    @staticmethod
    def generate_square_subsequent_mask(self, sz):
        """
        Hide all subsequent items because we can't see the future: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        :param sz: tensor to be masked
        :return mask: yo mask
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask):
        embedded = self.embedding(x)*math.sqrt(self.embeddingSize) #why?
        net = self.positionalencoding1d(embedded)
        net = self.encoder(net, mask)
        net = self.decoder(net)

        return net

# replier = Transformer()
# optimizer = optimizer.Adam(replier.parameters(), lr=3e-3)

#### Data Prep ####
with open("./trump_toys.csv", "r") as dataFile:
    csvReader = csv.reader(dataFile)
    dataset_raw = [i[4:] for i in csvReader]

dataset_x_raw = [deEmojify(i[0]) for i in dataset_raw]
dataset_y_raw = [deEmojify(i[1]) for i in dataset_raw]

tokenizer = get_tokenizer("basic_english")

dataset_x_tokenized = [tokenizer(i) for i in dataset_x_raw][1:]
dataset_y_tokenized = [tokenizer(i) for i in dataset_y_raw][1:]

dataset_pairwise = list(zip(dataset_x_tokenized, dataset_y_tokenized))

print(dataset_pairwise)

# print(dataset_x_tokenized)

