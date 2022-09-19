#!/usr/bin/env python
# coding: utf-8

# In[5]:



# python liberaries 

import numpy as np
import pandas as pd
import datetime, copy, imp
import time
import os
#import re
#import matplotlib.pyplot as plt
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


# In[6]:


# preprocess the data(join state and event tables, fill the null values with zero, add windowdiff column)
print("---------------------------------------")
print("Preprocessing, deleting inconsistency, join state and event tables, fill the null values with zero, add windowdiff column")
print("---------------------------------------")
tbl, state,events=pp.pre_processing(data_event_loc,data_state_loc)
tbl.to_hdf('data/tbl.hd5',key='Table')
state.to_hdf('data/state.hd5',key='State')
events.to_hdf('data/events.hd5',key='Events')

