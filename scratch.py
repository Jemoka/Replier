#pylint: disable=E1101

import re
import csv
import math
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

#### Utilities ####
np2tens = lambda x:torch.from_numpy(x).long()

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
    def __init__(self, numberTokens, embeddingSize, maxLength, numberEncoderLayers, numberDecoderLayers, attentionHeadCount, transformerHiddenDenseSize, batch_size=32):
        # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        super(Transformer, self).__init__()
        self.batch_size=batch_size
        self.model_type = 'Transformer'
        self.embeddingSize = embeddingSize
        self.numberTokens = numberTokens

        self.encoderEmbedding = nn.Embedding(numberTokens, embeddingSize)
        self.decoderEmbedding = nn.Embedding(numberTokens, embeddingSize)
        self.maxLength = maxLength 
        
        encoderLayer = nn.TransformerEncoderLayer(embeddingSize, attentionHeadCount, transformerHiddenDenseSize)

        self.encoder = nn.TransformerEncoder(encoderLayer, numberEncoderLayers)

        
        decoderLayer = nn.TransformerDecoderLayer(embeddingSize, attentionHeadCount)


        self.decoder = nn.TransformerDecoder(decoderLayer, numberDecoderLayers)

        self.decoderLinear = nn.Linear(embeddingSize, numberTokens)
        self.decoderSoftmax = nn.Softmax()

    @staticmethod
    def positionalencoding1d(d_model, length_max):
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
    def generate_square_subsequent_mask(sz):
        """
        Hide all subsequent items because we can't see the future: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        :param sz: tensor size to be masked
        :return mask: yo mask
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask, decoder_seed):
        embedded = self.encoderEmbedding(x)*math.sqrt(self.embeddingSize) #why?

        positional_encoding = self.positionalencoding1d(self.embeddingSize, self.maxLength)
        encoding_memory = self.encoder(positional_encoding+embedded, mask)

        seed_embedded = self.decoderEmbedding(decoder_seed)*math.sqrt(self.embeddingSize) #why?
        seed_positionalencoding = self.positionalencoding1d(self.embeddingSize, 1)

        seed = seed_embedded+seed_positionalencoding

        result = torch.Tensor()

        for _ in range(self.maxLength):
            net = self.decoder(seed, encoding_memory, tgt_mask=mask)
            net_decoded = self.decoderSoftmax(self.decoderLinear(net))

            result = torch.cat((result, net_decoded), 1)

            net_embeded = self.decoderEmbedding(torch.argmax(net_decoded, dim=2))*math.sqrt(self.embeddingSize) #why?
            net_positionalencoding = self.positionalencoding1d(self.embeddingSize, 1)

            seed = net_embeded+net_positionalencoding

        return result

# replier = Transformer()
# optimizer = optimizer.Adam(replier.parameters(), lr=3e-3)

#### Data Prep ####
with open("./trump_toys.csv", "r") as dataFile:
    csvReader = csv.reader(dataFile)
    dataset_raw = [i[4:] for i in csvReader]

dataset_x_raw = [deEmojify(i[0]) for i in dataset_raw]
dataset_y_raw = [deEmojify(i[1]) for i in dataset_raw]

tokenizer = get_tokenizer("basic_english")

vocabulary = defaultdict(lambda: len(vocabulary))

pad = vocabulary["<pad>"]
sos_token = vocabulary["<sos>"]
eos_token = vocabulary["<eos>"]

dataset_x_tokenized = [[vocabulary[e.lower().strip()] for e in tokenizer("<sos> "+i+" <eos>")] for i in dataset_x_raw][1:]
dataset_y_tokenized = [[vocabulary[e.lower().strip()] for e in tokenizer("<sos> "+i+" <eos>")] for i in dataset_y_raw][1:]

vocabulary_inversed = {v: k for k, v in vocabulary.items()}

max_length = max(max([len(i) for i in dataset_x_tokenized]), max([len(i) for i in dataset_y_tokenized]))

if max_length % 2 != 0:
    max_length += 1

dataset_x_padded = [x+(max_length-len(x))*[0] for x in dataset_x_tokenized]
dataset_y_padded = [y+(max_length-len(y))*[0] for y in dataset_y_tokenized]

# normalized_data = [list(zip(inp,oup)) for inp, oup in zip(dataset_x_tokenized, dataset_y_tokenized)] # pair up the data

batch_size = 8

chunk = lambda seq,size: list((seq[i*size:((i+1)*size)] for i in range(len(seq)))) # batchification

inputs_batched = np.array([i for i in chunk(dataset_x_padded, batch_size) if len(i) == batch_size]) # batchify and remove empty list
outputs_batched = np.array([i for i in chunk(dataset_y_padded, batch_size) if len(i) == batch_size]) # batchify and remove empty list


# inputs_batched = [np.array([np.array([e[0] for e in sentence]) for sentence in batch]) for batch in batches] # list of inputs
# outputs_batched = [np.array([np.array([e[1] for e in sentence]) for sentence in batch]) for batch in batches] # list of outputs


# inputs_batched = [] # list of onehot inputs
# outputs_batched = [] # list of onehot outputs


# for i in batches:
    # input_batch = [] # list of onehot inputs
    # output_batch = [] # list of onehot outputs
    # for e in i:
        # input_onehot = np.zeros(len(vocabulary))
        # input_onehot[e[0]] = 1
        # input_batch.append(input_onehot)
        # output_onehot = np.zeros(len(vocabulary))
        # output_onehot[e[1]] = 1
        # output_batch.append(output_onehot)
    # inputs_batched.append(np.array(input_batch))
    # outputs_batched.append(np.array(output_batch))

#### Hyperparametres ####
model = Transformer(len(vocabulary), maxLength=max_length, embeddingSize=200, numberEncoderLayers=4, numberDecoderLayers=4, attentionHeadCount=2, transformerHiddenDenseSize=200, batch_size=batch_size)

loss = nn.CrossEntropyLoss(reduction='sum')
lr = 5 # apparently Torch people think this is a good idea
adam = optimizer.Adam(model.parameters(), lr)

#### Training ####
epochs = 5
reporting = 4

model.train() # duh
mask = model.generate_square_subsequent_mask(batch_size)
for epoch in range(epochs):
    for batch, (inp, oup) in enumerate(zip(inputs_batched, outputs_batched)):
        breakpoint()
        inp_torch = np2tens(inp)
        oup_torch = np2tens(oup)

        adam.zero_grad()

        decoder_seed = torch.Tensor([[0]]*batch_size).type(torch.LongTensor)
        prediction = model(inp_torch, mask, decoder_seed)

        loss_val = loss(prediction.permute(0,2,1), oup_torch)

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        adam.step()

    print(f'| Epoch: {epoch} | Loss: {loss_val} |')

