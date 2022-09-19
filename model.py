#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import datetime, copy, imp
import time
import os
import re
import matplotlib.pyplot as plt
# progress bar
import tqdm
from tqdm.auto import tqdm, trange
from tqdm.notebook import tqdm
tqdm.pandas()
from datetime import timedelta
import copy
import sys
import gensim
from gensim.models.word2vec import Word2Vec
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder

# as we are willing to use the embeding layer in the first layer of the model, 
# we will use the Gensim liberary to give the weights to the embeding layer 

def creat_vectors_for_training(train_sequence_session,vector_size):
    model = gensim.models.Word2Vec(sentences=train_sequence_session, vector_size=vector_size)

    word_vectors_for_training = pd.np.insert(
        model.wv.vectors,   
        0, 
        pd.np.zeros(vector_size),
        axis=0
    )
    
    word_vectors_for_training = torch.FloatTensor(word_vectors_for_training)
    return(word_vectors_for_training)

# defining the model with differnt hidden layer and neuton per layer.
class Word2vec_NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_hiddin_layer,num_neurons_per_layer, num_classes,word_vectors_for_training,session_size, embedding_dim,mode):
        super().__init__()
        
        self.embedding_layer = nn.EmbeddingBag.from_pretrained(word_vectors_for_training,mode=mode)     
        self.layer1 = nn.Linear(input_size, num_neurons_per_layer)
        self.relu = nn.ReLU()
        self.num_hiddin_layer=num_hiddin_layer
        if num_hiddin_layer>1:
            self.fcs = nn.ModuleList()
            for i in range(num_hiddin_layer):
                self.fcs.append(nn.Linear(num_neurons_per_layer, num_neurons_per_layer))
            
        self.layer2 = nn.Linear(num_neurons_per_layer, num_classes)
        self.input_size=input_size
        self.embedding_dim=embedding_dim
        
    def forward(self, data_input):
        
        embedded_data_input = self.embedding_layer(data_input) 
        
        x = self.layer1(embedded_data_input)
        
        x = self.relu(x)
        if self.num_hiddin_layer>1:
            for l in self.fcs:
                x = F.relu(l(x))

        output = self.layer2(x)
        
        log_ps = F.log_softmax(output, dim=1)
 
        return log_ps

