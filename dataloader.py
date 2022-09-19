#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from torch.utils.data.sampler import WeightedRandomSampler


# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703

# Creating custom ways to order, batch and combine data with PyTorch DataLoaders and Random Sampler technique
def build_train_test_loader(x_train,y_train, x_test,y_test,batch_size):
    x_train=x_train.numpy()
    y_train=np.array(y_train).squeeze()
    x_test=x_test.numpy()
    y_test=np.array(y_test).squeeze()

    bs = batch_size

    class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
    print(class_sample_count)
    weight = 1. / class_sample_count
    weight_1 = class_sample_count*weight
    print(weight)

    samples_weight = np.array([weight[t-1] for t in y_train])
    samples_weight_1 = np.array([weight_1[t-1] for t in y_train])


    samples_weight = torch.from_numpy(samples_weight)
    samples_weight_1 = torch.from_numpy(samples_weight_1)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    sampler_1 = WeightedRandomSampler(samples_weight_1.type('torch.DoubleTensor'), len(samples_weight_1))

    # Creating PyTorch Datasets
    # datasets in pytorch have a length and are indexable So that:
    # 1- len(dataset) will work
    # 2- dataset[index] will return a tuple of (x,y)

    trainDataset = torch.utils.data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train.astype(int)))
    validDataset = torch.utils.data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test.astype(int)))

    # By wrappting datasets in a DataLoader, the data will become tesnors and we can iterate the datasets
    # we can use DataLoaders handy configurations like shuffling, batching, multi-processing

    # Every DataLoader has a Sampler which is used internally to get the indices for each batch. 
    # Each index is used to index into Dataset to grab the data (x, y). 

    # Using WeightedRandomSampler with a dataloader will build batches by randomly sampling from 
    # training set with the defined weight.
    # as we have given the highest weight to the classes with the fewere number of classes, 
    # with in each batch the number of classes are evently distrubuted. 

    trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=batch_size, num_workers=1, sampler = sampler)
    trainLoader_1 = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=batch_size, num_workers=1, sampler = sampler_1)

    testLoader = torch.utils.data.DataLoader(dataset = validDataset, batch_size=1, shuffle=False, num_workers=1) 


    print("---------------------------------------")
    print ('target train classes are hugely inbalanced as follows:')
    print(np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)]))
    print("---------------------------------------")
    print ('So, in each batch we have inbalanced classes as well, for example:')
    for i, (data, target) in enumerate(trainLoader_1):
        print("In batch index {} we have: ".format(i))
        for j in np.unique(target):
            print(len(np.where(target.numpy()==j)[0]),end="/")

        print("\n")
        break

    print("---------------------------------------")
    print ('This issue has been fixed with Weighted Random Sampler technique:')
    print("---------------------------------------")
    print ('So, in each batch we have balanced classes:')
    for i, (data, target) in enumerate(trainLoader):
        print("In batch index {} we have: ".format(i))
        for j in np.unique(target):
            print(len(np.where(target.numpy()==j)[0]),end="/")

        print("\n")
        break
    print("---------------------------------------")
    
    return(trainLoader, testLoader )


# In[ ]:




