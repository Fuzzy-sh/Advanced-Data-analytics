#!/usr/bin/env python
# coding: utf-8

# # Analysis Start Point
# This notebook loads a version of the MLB data that meant to represent medical data.  This data contains:
# 1. A series of discrete events: 
#   + `GoodTestResult`: A medical test that comes back with good results.
#   + `BadTestResult`: A medical test that comes back with bad results.
#   + `VitalsCrash`: A very bad event where the individuals vitals crash (ie. a heart attack) and emergency medical aid is required.
#   
# 2. A series of events that stretch over a period of time and have a starting and ending point:
#   + `StartAntibiotics/EndAntibiotics`: An individual starts/ends a course of very strong antibiotics.
#   + `StartHospital/EndHospital`: An individual starts/ends a stay in hospital.
#   + `StartIcu/EndIcu`: An individual starts/ends a stay in the ICU.

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '1')


# # import the liberaries

# In[6]:



# python liberaries 

import numpy as np
import pandas as pd
import datetime, copy, imp
import time
import os
import re
import matplotlib.pyplot as plt
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

# importing our defined .py packeges file as a calss
import preprocessing as pp
import dataloader as dl
import model as mdl
import train as tr

# defining the location of reading and writing files 
data_event_loc='data/MLB-MedicalEvents.hd5'
data_state_loc='data/MLB-MedicalState.hd5'
dir_checkpoints='model/'
dir_checkpoints="model/"
dir_results='results/'

# if there is gpu available, train the model in the gpu
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------------------

# difining the hyperparameters 
# hyperparameters with constant values

epochs = 1500
print_every=100
pading_num=0
window_size_min=-1

# ------------------------------
# hyperparameters with variaty of values

params = sys.argv[1:]
window_size_max=int(params[0]) # 30,40,50,60

train_test_split_rate=float(params[1]) # 0.80, 0.90
batch_size=int(params[2]) # 16,32,64
lr=float(params[3]) # 0,001, 0.0001
num_neurons_per_layer=int(params[4]) # 16,32, 64,128,256
num_hiddin_layer=int(params[5]) # 1,2,3
mode=params[6] # 'mean' , 'max', 'sum'


# In[ ]:





# ## Load Data and preprocessing 
# 

# In[37]:


# AFter preprocessing the data(join state and event tables, fill the null values with zero, add windowdiff column) 
# we will read the data 

tbl=pd.read_hdf('data/tbl.hd5',key='Table')
state=pd.read_hdf('data/state.hd5',key='State')
events=pd.read_hdf('data/events.hd5',key='Events')


# In[38]:



# Add session column to the table
print("Adding Session to the table by window size of {} ".format(window_size_max))
print("---------------------------------------")
tbl=pp.Add_session_defined_by_windowsize(tbl,window_size_max,window_size_min)

print("Creating a table for the person ... ")
print("---------------------------------------")
# Creating a table for the person to split the database into train and test, based on the paitents
person_tbl=pd.DataFrame(tbl.index.get_level_values(0).unique())
# split the person_table by 90% for train and 10% for test.

print("train test split rate for training is : {}.".format(train_test_split_rate))
print("---------------------------------------")
person_tbl_train, person_tbl_test = model_selection.train_test_split(person_tbl, train_size=train_test_split_rate, random_state=101)

# split train and test
tbl_train,tbl_test =pp.train_test_split_tbl(tbl,person_tbl,person_tbl_train,person_tbl_test)

print("---------------------------------------")
print( "All the rows are: {}".format(len(tbl)))
print("---------------------------------------")
print("The lengh of the table with sessions involved: {}".format(len(tbl[tbl['sessionId']!=0])))
print("---------------------------------------")
# print("So, {} number of rows do not have start and end of hospital".format(len(tbl)-len(tbl_[tbl_['sessionId']!=0])))

print("The lengh of the train_table: {}".format(len(tbl_train)))
print("---------------------------------------")
print("The lengh of the test_table: {}".format(len(tbl_test)))   

#Building tables for sessions, event, and session_event table to relate these two tables
event_tbl=pp.build_event_tbl(tbl)

# for each train and test tabel, we need related session table 

train_session_tbl=pp.build_session_table(tbl_train)
test_session_tbl=pp.build_session_table(tbl_test)
print("---------------------------------------")
print("The lengh of the train_session_tbl: {}".format(len(train_session_tbl)))
print("---------------------------------------")
print("The lengh of the test_session_tbl: {}".format(len(test_session_tbl)))
print("---------------------------------------")

# we have calculated idx to set this column as the new index
train_session_event_tbl=pp.build_session_event_table(tbl_train,event_tbl)
test_session_event_tbl=pp.build_session_event_table(tbl_test,event_tbl)
print("The lengh of the train_session_event_tbl: {}".format(len(train_session_event_tbl)))
print("---------------------------------------")
print("The lengh of the test_session_tbl: {}".format(len(test_session_tbl)))
print("---------------------------------------")

# for each train and test tabel, we need related session_event table 
train_session_event_tbl_=pp.remove_session_with_len_lessthan_three(train_session_event_tbl)
test_session_event_tbl_=pp.remove_session_with_len_lessthan_three(test_session_event_tbl)

print("The lengh of the train_session_event_tbl: {}".format(len(train_session_event_tbl_)))
print("---------------------------------------")
print("The lengh of the test_session_event_tbl: {}".format(len(test_session_event_tbl_)))  
print("---------------------------------------")

print("Creating sequence of the sessions for both training and testing session event tables..")
print("---------------------------------------")
train_sequence_session=pp.creat_session_sequence(train_session_event_tbl_)
test_sequence_session=pp.creat_session_sequence(test_session_event_tbl_)

print("Creating X and Y for each session for both training and testing seqience session tables..")
print("---------------------------------------")
x_train,y_train=pp.x_y_split(train_sequence_session)
x_test,y_test=pp.x_y_split(test_sequence_session)

print("Adding zero padding to the end of the sequence of the data to reach to the max length.")
print("---------------------------------------")
x_train,x_test,vector_size_for_train=pp.padding_the_sequence(x_train,x_test,pading_num)


# ## visualizaion on the classes 

# In[ ]:


# to count each classes of the dataset and visualize them 
df_y_train=pd.DataFrame(y_train,columns=['count'])
y_train_count=pd.DataFrame(df_y_train['count'].value_counts())
y_train_count.reset_index(inplace=True)

y_train_count.rename(columns={'index': 'eventId'},inplace=True)

y_train_count=pd.merge(y_train_count,event_tbl,on='eventId',how='inner')

# do some visualizaion for the classes to see if the classes are imbalanced

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.bar(y_train_count['event'],y_train_count['count'])
plt.ylabel('Count of each classes')
plt.xlabel('classes')
plt.show()


# ## Bulding the data loaders
# 
# 
# 

# In[ ]:


# Bulding a trainloader and testLoader, by which we can easily iterate through the database by batches. 


# In[45]:


trainLoader, testLoader = dl.build_train_test_loader(x_train,y_train, x_test,y_test,batch_size)


# # Configurable neural network
# We can only tune those parameters that are configurable,like  layer sizes of the fully connected layers:

# In[ ]:


import tqdm
import tqdm.notebook as tq
################################


###############################
# Define the network

word_vectors_for_training=mdl.creat_vectors_for_training(train_sequence_session,vector_size_for_train)
session_size, embedding_dim=word_vectors_for_training.numpy().shape
# session_size, embedding_dim=(9,10)
data, target =next(iter(trainLoader))
num_b, input_size=data.numpy().shape
num_classes=9


word2vec_network = mdl.Word2vec_NeuralNetwork(input_size, num_hiddin_layer,num_neurons_per_layer, num_classes,word_vectors_for_training,session_size, embedding_dim, mode)

for params in word2vec_network.parameters():
    params.required_grad=True
    


optimizer = optim.SGD(word2vec_network.parameters(), lr=lr)
loss_function = nn.NLLLoss()
word2vec_network.to(device)


tr.train_model(epochs, print_every,trainLoader,testLoader,optimizer,loss_function,word2vec_network,window_size_max,batch_size,num_neurons_per_layer,num_hiddin_layer,train_test_split_rate,lr,device,dir_checkpoints,dir_results,mode)

# model = mdl.Word2vec_NeuralNetwork(input_size, num_hiddin_layer,num_neurons_per_layer, num_classes,word_vectors_for_training,session_size, embedding_dim, mode)
# tr.test_model(10,dir_checkpoints+"word2vec_network_w_60_b_32_nn_64_nh2.pth",input_size, num_hiddin_layer,num_neurons_per_layer, num_classes,word_vectors_for_training,session_size, embedding_dim,model,testLoader,loss_function,event_tbl,pading_num)

