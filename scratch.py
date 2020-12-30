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
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer

import matplotlib
import matplotlib.pyplot as plt
from  bpe import Encoder
from matplotlib.lines import *

# matplotlib.use('pdf')  # Or any other X11 back-end

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

## util to check gradient flow from https://github.com/alwynmathew/gradflow-check
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    print([i.item() for i in ave_grads])
#     plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.ion()
    # plt.show()
    
def plot_grad_flow_bars(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.legend([Line2D([0], [0], color="c", lw=4),
                # Line2D([0], [0], color="b", lw=4),
                # Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # plt.show()

#### Network ####
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class Transformer(nn.Module):
    # def __init__(self, numberTokens:int, embeddingSize:int, attentionHead:int, hiddenDenseSize:int, numberLayers:int):
    def __init__(self, numberTokens, embeddingSize, maxLength, numberEncoderLayers, numberDecoderLayers, attentionHeadCount, transformerHiddenDenseSize, linearHiddenSize):
        # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        super(Transformer, self).__init__()
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

        self.outputHidden = nn.Linear(embeddingSize, linearHiddenSize);
        self.outputHidden2 = nn.Linear(linearHiddenSize, linearHiddenSize)
        self.outputLayer = nn.Linear(linearHiddenSize, numberTokens)
        self.outputSoftmax = nn.Softmax(dim=2)


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



    def forward(self, x, raw_seed=None, batch_size=32):
        if raw_seed != None:
            decoder_seed = raw_seed
        else:
            decoder_seed = torch.Tensor([[1]]*batch_size).type(torch.LongTensor).cuda()

        embedded = self.encoderEmbedding(x)*math.sqrt(self.embeddingSize) #why?

        positional_encoding = self.positionalencoding1d(self.embeddingSize, self.maxLength).cuda()
        encoder_input = embedded+positional_encoding

        encoder_padding_mask = torch.eq(x, 0)
        encoder_memory = self.encoder(encoder_input.transpose(0,1), src_key_padding_mask=encoder_padding_mask)
    
        # decoder_seed = 
        seed = self.decoderEmbedding(decoder_seed)

        decoder_memory = seed

        if self.training:
            positional_encoding = self.positionalencoding1d(self.embeddingSize, self.maxLength).cuda()
            decoder_input = positional_encoding + decoder_memory

            decoder_mask = self.generate_square_subsequent_mask(self.maxLength).cuda()
            decoder_padding_mask = torch.eq(decoder_seed, 0)

            net = self.decoder(decoder_input.transpose(0,1), encoder_memory, tgt_mask=decoder_mask, tgt_key_padding_mask=decoder_padding_mask, memory_key_padding_mask=encoder_padding_mask).transpose(0,1)

        else:
            for i in range(self.maxLength):
                positional_encoding = self.positionalencoding1d(self.embeddingSize, i+1).cuda()
                decoder_input = positional_encoding + decoder_memory

                decoder_mask = self.generate_square_subsequent_mask(i+1).cuda()

                net = self.decoder(decoder_input.transpose(0,1), encoder_memory, tgt_mask=decoder_mask).transpose(0,1)

                decoder_memory = torch.cat((seed, net), dim=1)

        net = self.outputHidden(net)
        net = self.outputHidden2(net)
        net = self.outputLayer(net)
        net = self.outputSoftmax(net)

        return net

# replier = Transformer()
# optimizer = optimizer.Adam(replier.parameters(), lr=3e-3)

#### Data Prep ####
dataset_name = "./movie_replies.csv"

with open(dataset_name, "r") as dataFile:
    csvReader = csv.reader(dataFile, delimiter="±")
    dataset_raw = [i[4:] for i in csvReader]

dataset_x_raw = [deEmojify(i[0]) for i in dataset_raw]
dataset_y_raw = [deEmojify(i[1]) for i in dataset_raw]

# <<<<<<< HEAD
zipped_dataset = list(zip(dataset_x_raw, dataset_y_raw))

# # crop the dataset b/c we don't have the big bucks
zipped_dataset = zipped_dataset[-15000:]
# =======
# zipped_dataset = list(zip(dataset_x_raw, dataset_y_raw))
# >>>>>>> c252b6a881ae62cf53b15440272c4567a7aea0b2

dataset_x_raw, dataset_y_raw = zip(*zipped_dataset)

tokenizer = get_tokenizer("revtok")
# tokenizer = Encoder(12500, pct_bpe=0.88)  # params chosen for demonstration purposes
# tokenizer.fit(dataset_x_raw+dataset_y_raw)

vocabulary = defaultdict(lambda: len(vocabulary))

pad = vocabulary["__pad"]
sos_token = vocabulary["__sos"]
eos_token = vocabulary["__eos"]
# sow_token = vocabulary["__sow"]
# eow_token = vocabulary["__eow"]


dataset_x_tokenized = [[sos_token]+[vocabulary[e.lower().strip()] for e in tokenizer(i)]+[eos_token] for i in dataset_x_raw][1:]
dataset_y_tokenized = [[sos_token]+[vocabulary[e.lower().strip()] for e in tokenizer(i)]+[eos_token] for i in dataset_y_raw][1:]

vocabulary_inversed = {v: k for k, v in vocabulary.items()}

max_length = max(max([len(i) for i in dataset_x_tokenized]), max([len(i) for i in dataset_y_tokenized]))+2 # +2 for safety ig

if max_length % 2 != 0:
    max_length += 1

dataset_x_padded = [x+(max_length-len(x))*[0] for x in dataset_x_tokenized]
dataset_y_padded = [y+(max_length-len(y))*[0] for y in dataset_y_tokenized]

# normalized_data = [list(zip(inp,oup)) for inp, oup in zip(dataset_x_tokenized, dataset_y_tokenized)] # pair up the data

batch_size = 64

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


#### Test Sentence Prep ####
sentences = ["I am a smart.", "He is very smart"];
prediction_batch_size = len(sentences);

# prediction_x_tokenized = [[vocabulary[e.lower().strip()] for e in tokenizer(i+" <eos>")] for i in sentences]
prediction_x_tokenized = [[sos_token]+[vocabulary[e.lower().strip()] for e in tokenizer(i)]+[eos_token] for i in sentences]
# dataset_y_tokenized = [[sos_token]+[vocabulary[e.lower().strip()] for e in tokenizer(i)]+[eos_token] for i in dataset_y_raw][1:]


prediction_x_padded = np.array([x+(max_length-len(x))*[0] for x in prediction_x_tokenized])

prediction_x_torch = np2tens(prediction_x_padded).transpose(0,1)

#### Hyperparametres ####
# <<<<<<< HEAD
# model = Transformer(4081, maxLength=max_length, embeddingSize=128, numberEncoderLayers=4, numberDecoderLayers=4, attentionHeadCount=8, transformerHiddenDenseSize=256)

# =======
model = nn.DataParallel(Transformer(len(vocabulary), maxLength=max_length, embeddingSize=258, numberEncoderLayers=2, numberDecoderLayers=2, attentionHeadCount=6, transformerHiddenDenseSize=512, linearHiddenSize=512).cuda())
# >>>>>>> c252b6a881ae62cf53b15440272c4567a7aea0b2

# Weight Initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.005)
    elif type(m) == nn.LayerNorm:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0.005)
model.apply(init_weights)

# def crossEntropy(logits, targets_sparse, epsilon=1e-8):
    # targets = nn.functional.one_hot(targets_sparse, len(vocabulary))
    # target_mask = torch.not_equal(targets_sparse, 0).float()
    # cross_entropy = torch.mean(-torch.log(torch.gather(logits+targets, 1, targets).squeeze(1)), -1)
    # return torch.mean(target_mask*cross_entropy)

# def crossEntropy(logits, targets_sparse, epsilon=1e-8):
    # targets = nn.functional.one_hot(targets_sparse, len(vocabulary))
    # target_mask = torch.not_equal(targets_sparse, 0).float()

    # loss_vals = torch.sum(- targets * F.log_softmax(logits+epsilon, -1), -1)

    # return torch.mean(target_mask*loss_vals)
        

crossEntropy = torch.nn.CrossEntropyLoss(reduce=False)
def maskedCrossEntropy(logits, targets_sparse):
    vals = crossEntropy(logits.transpose(1,2), targets_sparse)
    target_mask = torch.not_equal(targets_sparse, 0).float()
    return torch.mean(target_mask*vals)

# nll = torch.nn.NLLLoss(reduce=False)
# def maskeddNLL(logits, targets_sparse):
    # vals = nll(logits.transpose(1,2), targets_sparse)
    # target_mask = torch.not_equal(targets_sparse, 0).float()
    # return torch.mean(target_mask*vals)


criterion = maskedCrossEntropy
lr = 0.25 # apparently Torch people think this is a good idea
# apparently Torch people think this is a good idea
adam = optimizer.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[2,20], gamma=0.75) # decay schedule

#### Training ####
def training(retrain=None):
    if retrain is not None:
        checkpoint = torch.load(retrain, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])

    epochs = 100000
    reporting = 2
    # accumulate = 24
    accumulate = 40
    print(f'Effective batch size: {batch_size*accumulate}')

    version = "DEC212020_1_HUGELR"
    modelID = str(uuid.uuid4())[-5:]
    initialRuntime = time.time()

    writer = SummaryWriter(f'./training/movie/logs/{modelID}')

# random.shuffle(zipped_dataset)
    

    model.train() # duh
    for epoch in range(epochs):
        total_loss = 0
        #
#         if (epoch % 3 == 0) and epoch != 0:
            # print(f'Taking a 15 min fridge break before starting at {epoch}...')
            # for _ in tqdm(range(60*15)):
                # time.sleep(1)
            # print(f'Fridge break done. Let\'s get cracking on epoch {epoch}')

        checkpointID = str(uuid.uuid4())[-5:]
        batch_data_group = list(zip(inputs_batched, outputs_batched))

        random.shuffle(batch_data_group)

        batch_data_feed = tqdm(enumerate(batch_data_group), total=len(inputs_batched))

        for batch, (inp, oup) in batch_data_feed:
            encinp_torch = np2tens(inp).cuda()
            decinp_torch = np2tens(oup).cuda()

            padding_row = torch.zeros(batch_size,1)
            oup_torch = (torch.cat((np2tens(oup)[:, 1:], padding_row), dim=1)).long().cuda()

            # decInp_torch

            prediction = model(encinp_torch, decinp_torch, int(batch_size/2))

            loss_val = criterion(prediction, oup_torch)/accumulate
            total_loss += loss_val.item()

#             target_mask = torch.not_equal(oup_torch, 0).float()
            # loss_matrix = torch.mean((prediction-torch.nn.functional.one_hot(oup_torch, len(vocabulary)))**2, 2)
            # loss_val = torch.mean(target_mask*loss_matrix)
            loss_val.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            # plot_grad_flow(model.named_parameters())
            if ((batch+(epoch*len(inputs_batched)))%accumulate) == 0 and batch != 0:
                adam.step()
                adam.zero_grad()

            prediction_values = np.array(torch.argmax(prediction,2).cpu())[:1]
        
            prediction_sentences = []
            for e in prediction_values:
                prediction_value = []
                for i in e:
                    try: 
                        prediction_value.append(vocabulary_inversed[i])
                    except KeyError:
                        prediction_value.append("__err")
                prediction_sentences.append(prediction_value)

            final_sent = ""
            for word in prediction_sentences[0]:
                final_sent = final_sent + word + " "

            # ont = list(model.named_parameters())
            # breakpoint()

            loss_val_avg = total_loss/(batch+1)

            # breakpoint()

            writer.add_scalar('Train/loss', loss_val.item(), batch+(epoch*len(inputs_batched)))
            writer.add_scalar('Train/avgloss', loss_val_avg, batch+(epoch*len(inputs_batched)))
            writer.add_text('Train/sample', final_sent, batch+(epoch*len(inputs_batched)))


            # plot_grad_flow(model.named_parameters())
            # breakpoint()
            
            batch_data_feed.set_description(f'| Model: {modelID}@{checkpointID} | Epoch: {epoch} | Batch: {batch} | Avg Loss: {loss_val_avg:.5f} |')
        #plot_grad_flow(model.named_parameters())

        # CheckpointID,ModelID,ModelVersion,Dataset,Initial Runtime,Current Time,Epoch,Loss,Checkpoint Filename

        initialHumanTime = datetime.fromtimestamp(initialRuntime).strftime("%m/%d/%Y, %H:%M:%S")
        nowHumanTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        writer.add_scalar('Train/epochloss', loss_val_avg, (epoch*len(inputs_batched)))

        scheduler.step()
        with open("./training/movie/training-log.csv", "a+") as df:
            csvfile = csv.writer(df)
            csvfile.writerow([checkpointID, modelID, version, dataset_name, initialHumanTime, nowHumanTime, epoch, loss_val.item(), f'{modelID}-{checkpointID}.model', f'{retrain}'])

        torch.save({
            'version': version,
            'modelID': modelID,
            'checkpointID': checkpointID,
            'datasetName': dataset_name,
            'epoch': epoch,
            'loss': loss_val,
            'model_state': model.state_dict(),
            'optimizer_state': adam.state_dict(),
            'lr': scheduler.get_lr()
            }, f'./training/movie/{modelID}-{checkpointID}.model')

        print(f'| EPOCH DONE | Epoch: {epoch} | Avg Loss: {loss_val_avg} |')
    writer.close()

def inferring(url):
    checkpoint = torch.load(url, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

                
    with torch.no_grad():
        mask = model.generate_square_subsequent_mask(max_length)
        start_flush = torch.Tensor([[1]*prediction_batch_size]).type(torch.LongTensor)
        prediction = model(prediction_x_torch, start_flush, mask, prediction_batch_size)

        prediction_values = np.array(torch.argmax(prediction,2).transpose(0,1))
        
    prediction_sentences = [[vocabulary_inversed[i] for i in e] for e in prediction_values]
    breakpoint()

training()

# training()


