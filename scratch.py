#pylint: disable=E1101

import re
import csv
import time
import uuid
import math
import random
import numpy as np
from datetime import datetime

from tqdm import tqdm

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer

import matplotlib.pyplot as plt

#### Utilities ####
# util to tenserify them numpy arrays
np2tens = lambda x:torch.from_numpy(x).long()

# util to get rid of emojis
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

## util to check gradient flow from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

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
        self.maxLength = maxLength 
        
        encoderLayer = nn.TransformerEncoderLayer(embeddingSize, attentionHeadCount, transformerHiddenDenseSize)

        self.encoder = nn.TransformerEncoder(encoderLayer, numberEncoderLayers)


        self.decoderEmbedding = nn.Embedding(numberTokens, embeddingSize)
        
        decoderLayer = nn.TransformerDecoderLayer(embeddingSize, attentionHeadCount, transformerHiddenDenseSize)

        self.decoder = nn.TransformerDecoder(decoderLayer, numberDecoderLayers)

        self.decoderLinear = nn.Linear(embeddingSize, numberTokens)
        self.decoderSoftmax = nn.Softmax(dim=2)


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



    def forward(self, x, decoder_seed, mask):

        embedded = self.encoderEmbedding(x)*math.sqrt(self.embeddingSize) #why?

        positional_encoding = self.positionalencoding1d(self.embeddingSize, self.batch_size)
        encoding_memory = self.encoder(positional_encoding+embedded, mask)

        seed = self.decoderEmbedding(decoder_seed)

        decoder_input = seed

        for i in range(self.maxLength):
            positional_encoding = self.positionalencoding1d(self.embeddingSize, self.batch_size)
            decoder_input_pos = positional_encoding + decoder_input
            net = self.decoder(decoder_input_pos, encoding_memory, tgt_mask=self.generate_square_subsequent_mask(i+1))
            decoder_input = torch.cat((seed, net), dim=0)
        
        return self.decoderSoftmax(self.decoderLinear(net))

# replier = Transformer()
# optimizer = optimizer.Adam(replier.parameters(), lr=3e-3)

#### Data Prep ####
dataset_name = "./trump_replies.csv"

with open(dataset_name, "r") as dataFile:
    csvReader = csv.reader(dataFile)
    dataset_raw = [i[4:] for i in csvReader]

dataset_x_raw = [deEmojify(i[0]) for i in dataset_raw]
dataset_y_raw = [deEmojify(i[1]) for i in dataset_raw]

zipped_dataset = list(zip(dataset_x_raw, dataset_y_raw))
random.shuffle(zipped_dataset)

dataset_x_raw, dataset_y_raw = zip(*zipped_dataset)

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
model = Transformer(len(vocabulary), maxLength=max_length, embeddingSize=150, numberEncoderLayers=2, numberDecoderLayers=2, attentionHeadCount=2, transformerHiddenDenseSize=128, batch_size=batch_size)

def crossEntropy(logits, targets):
    return torch.mean(torch.sum(- targets * F.log_softmax(logits, -1), -1))

criterion = crossEntropy
lr = 1e-2 # apparently Torch people think this is a good idea
adam = optimizer.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(adam, 1.0, gamma=0.95) # decay schedule

#### Training ####
epochs = 100
reporting = 2

version = "NOV232020_1"
modelID = str(uuid.uuid4())[-5:]
initialRuntime = time.time()

model.train() # duh
mask = model.generate_square_subsequent_mask(max_length)
for epoch in range(epochs):
    checkpointID = str(uuid.uuid4())[-5:]
    batch_data_feed = tqdm(enumerate(zip(inputs_batched, outputs_batched)), total=len(inputs_batched))
    for batch, (inp, oup) in batch_data_feed:
        inp_torch = np2tens(inp).transpose(0,1)
        oup_torch = np2tens(oup).transpose(0,1)

        oup_oh = nn.functional.one_hot(oup_torch, len(vocabulary))

        start_flush = torch.Tensor([[1]*batch_size]).type(torch.LongTensor)
        decoder_seed = start_flush

        adam.zero_grad()

        seq = torch.Tensor()

        prediction = model(inp_torch, decoder_seed, mask)

        loss_val = criterion(prediction, oup_oh)
        loss_val.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        plot_grad_flow(model.named_parameters())
        breakpoint()

        adam.step()

        batch_data_feed.set_description(f'| Model: {modelID}@{checkpointID} | Epoch: {epoch} | Batch: {batch} | Loss: {loss_val:.2f} |')

    # CheckpointID,ModelID,ModelVersion,Dataset,Initial Runtime,Current Time,Epoch,Loss,Checkpoint Filename

    initialHumanTime = datetime.fromtimestamp(initialRuntime).strftime("%m/%d/%Y, %H:%M:%S")
    nowHumanTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open("./training/trump/training-log.csv", "a") as df:
        writer = csv.writer(df)
        writer.writerow([checkpointID, modelID, version, dataset_name, initialHumanTime, nowHumanTime, epoch, loss_val.item(), f'{modelID}-{checkpointID}.model'])

    torch.save({
        'version': version,
        'modelID': modelID,
        'checkpointID': checkpointID,
        'datasetName': dataset_name,
        'epoch': epoch,
        'loss': loss_val,
        'model_state': model.state_dict(),
        'optimizer_state': adam.state_dict(),
        'lr': scheduler.get_last_lr()
    }, f'./training/trump/{modelID}-{checkpointID}.model')

    print(f'| EPOCH DONE | Epoch: {epoch} | Loss: {loss_val} |')
    scheduler.step()

