#!/usr/bin/env python
# coding: utf-8

# In[2]:


###########################
# train the model function
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
import tqdm

def train_model(epochs, print_every,trainLoader,testLoader,optimizer,loss_function,word2vec_network,window_size_max,batch_size,num_neurons_per_layer,num_hiddin_layer,train_test_split_rate,lr,device,dir_checkpoints,dir_results,mode):
    
    # we need to have several lists for loss and accuracy results that we are calculating during the training
    results={'epoch':[],'training_loss':[], 'test_loss':[], 'test_accuracy':[], 'training_accuracy':[]}
    steps= 0
    min_val_loss = np.Inf
    
    # for each to iterate through the epochs and reset the acuracy and loss for each epoch
    for e in range(1,epochs+1):
        running_results={'loss':0, 'accuracy':0,'steps':0}
        # we set the model in the training phase, so the model can learn
        word2vec_network.train()
        # set the dataloader with the tqdm liberary so we can see the tqdm bar during the training 
        train_bar=tqdm.tqdm(trainLoader)

        
        # iterate through the trainloader for each batch
        for i, (inputs, labels) in enumerate(train_bar):
                # labels needs to be integer before fitting the model
                labels=torch.LongTensor(labels)
                
                # if we have the gpu availabel, we need to put both labels and input data into the availabel devices
                inputs, labels = inputs.to(device), labels.to(device)

                # increament the step that we are in
                running_results['steps']+=1
                # we need to remove the gradient graph that we have biult for the previous batches
                optimizer.zero_grad()
                # train the model 
                output=word2vec_network.forward(inputs)
                # as we return the log softmax, to recieve the amount of output we need the exp of the output. 
                ps=torch.exp(output)
                # claculate the loss based on the NLLLoss
                loss=loss_function(output,labels-1)
                # bakward training based on the loss 
                loss.backward()
                # optimize the model
                optimizer.step()
                # increament the loss as itterating through the batches.
                running_results['loss']+=loss.item()
                # claculate if the predicted value is the same as the label
                equality = (ps.max(dim=1)[1]==(labels-1).data)
                # claculate the mean of the acuracy for each back 
                running_results['accuracy']+=equality.type(torch.FloatTensor).mean()
                # add all of the information to the window, so the results can be followed diring the training 
                train_bar.set_description(desc='[%d/%d], training_loss: %.4f, training_accuracy: %.4f '% (e,epochs, running_results['loss']/running_results['steps'],running_results['accuracy']/running_results['steps']))
        
        # set the tqdm for the testloader
        test_bar=tqdm.tqdm(testLoader, desc='Validation Results:')
        # set the model in the evaluation phase, so the weight wont be changed while validating
        word2vec_network.eval()
        # the informaion for test loss and accutacy
        valing_results={'test_loss':0, 'test_accuracy':0, 'min_loss':0, 'steps':0}
        
        # with no grad, the model is prepared to be evaluated
        with torch.no_grad():
            # iterate through the test lader based on the batches
            for i, (inputs, labels) in enumerate(test_bar):
                # the explantion for the following cods are the same as the training and the main difference is in 
                # the evaluation phase, the wieghts do not change
                valing_results['steps']+=1

                labels=torch.LongTensor(labels)
                inputs, labels = inputs.to(device), labels.to(device)
                output=word2vec_network.forward(inputs)
                ps=torch.exp(output)
        
                loss=loss_function(output,labels-1)
                valing_results['test_loss']+= loss.item()
                equality = (ps.max(dim=1)[1]==(labels-1).data)
                valing_results['test_accuracy']+=equality.type(torch.FloatTensor).mean()
                test_bar.set_description(desc='test_loss: %.4f, test_accuracy: %.4f'% (valing_results['test_loss']/valing_results['steps'], valing_results['test_accuracy']/valing_results['steps'] ))
            
            # the model with the least test loss is suitable to be used for the new data 
            if (valing_results['test_loss']/valing_results['steps']< min_val_loss):
                    min_val_loss=valing_results['test_loss']/valing_results['steps']
                    print('saving the model with min loss of : '+ str(min_val_loss))
                    # we are saving the model and call them based on different hyper parameters
                    # window_size_max,batch_size,num_neurons_per_layer,num_hiddin_layer
                    torch.save(word2vec_network.state_dict(),dir_checkpoints+"w_{}_b_{}_nn_{}_nh{}_spl_{}_lr_{}_mod_{}.pth".format(window_size_max,batch_size,num_neurons_per_layer,num_hiddin_layer,train_test_split_rate,lr,mode))

        # save the results in the dictionary
        results['training_loss'].append(running_results['loss']/running_results['steps'])
        results['test_loss'].append(valing_results['test_loss']/valing_results['steps'])
        results['test_accuracy'].append(valing_results['test_accuracy'].item()/valing_results['steps'])
        results['training_accuracy'].append(running_results['accuracy'].item()/running_results['steps'])
        
        # the dictionary of the results will be saved to a csv file
        data_frame=pd.DataFrame(
            data={
                # 'Epoch':1,
                'Training_Loss':results['training_loss'],
                'Test_Loss': results['test_loss'],
                'Test_Accuracy':results['test_accuracy'],
                'Training_Accuracy':results['training_accuracy'],

                     },
            index=range(1,e+1)
        )
        
        data_frame.to_csv(dir_results+'{}_b_{}_nn_{}_nh{}_spl_{}_lr_{}_mod_{}.csv'.format(window_size_max,batch_size,num_neurons_per_layer,num_hiddin_layer,train_test_split_rate,lr,mode),  index_label="Epoch")    


# In[ ]:

# To see the 
def test_model(num_results, best_model,input_size, num_hiddin_layer,num_neurons_per_layer, num_classes,word_vectors_for_training,session_size, embedding_dim,model,testLoader,loss_function,event_tbl,pading_num):

    model.load_state_dict(torch.load(best_model))
    model.eval()

    test_bar=tqdm.tqdm(testLoader)

    k=0
    for i, (inputs, labels) in enumerate(test_bar):

        with torch.no_grad():
            output = model(inputs)
            ps=torch.exp(output)
            loss=loss_function(output,labels-1)
        print("---------------------------------------")
        for row in inputs.numpy():
            for i in row:
                if i!=pading_num:
                    print(event_tbl[event_tbl.eventId==i].event.values)
                else:
                    continue

        print("---------------------------------------")
        print("target should be:"+event_tbl[event_tbl.eventId==labels.numpy().squeeze()].event.values)
        print("---------------------------------------")


        predicted_tbl=pd.DataFrame(ps.detach().numpy()).T
        predicted_tbl['eventId']=event_tbl['eventId']
        predicted_tbl.set_index('eventId')

        predicted_results=pd.merge(predicted_tbl,event_tbl,on='eventId').sort_values(by=0,ascending=False)
        for r in zip((predicted_results[0]*100).round(),predicted_results['event']):
            print(r)
        k +=1
        if k==num_results:
            break  


