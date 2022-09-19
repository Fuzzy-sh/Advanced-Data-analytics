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

# In[416]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '1')


# In[417]:


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

import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data.sampler import WeightedRandomSampler

# from gensim.models import predict_


data_event_loc='data/MLB-MedicalEvents.hd5'
data_state_loc='data/MLB-MedicalState.hd5'


# ## Load Data and preprocessing 
# 

# In[418]:


def howOften_ends_comesAfter_Start(tbl, startState, endState):
    Start_needs_end_Flag=False
    End_needs_end_Flag=False
    inconsistancy_counter=0
    incosistancy_tbl=pd.DataFrame()

    
    
    for i_2,row in tbl.iterrows():
        if (row.Tag==startState) & (not Start_needs_end_Flag):
            Start_needs_end_Flag=True
            End_needs_end_Flag=False
            r=row
            i_1=i_2
            continue
            
        if (row.Tag==endState) & (not End_needs_end_Flag):
            Start_needs_end_Flag=False
            End_needs_end_Flag=True
            r=row
            i_1=i_2
            continue
            
        if (row.Tag==startState) & (Start_needs_end_Flag):
#             print(r.Tag,row.Tag)
            incosistancy_tbl=incosistancy_tbl.append(pd.DataFrame(i_1).T)
            
            r=row
            i_1=i_2
            inconsistancy_counter=inconsistancy_counter+1
            continue
            
        if (row.Tag==endState) & (End_needs_end_Flag):
#             print(r.Tag,row.Tag)
            incosistancy_tbl=incosistancy_tbl.append(pd.DataFrame(i_1).T)
            r=row
            i_1=i_2
            inconsistancy_counter=inconsistancy_counter+1
            continue
            
#         r=row
    howOften=(inconsistancy_counter/len(tbl))*100

#     print("{} rows out of {} state , the {} was happend before the {} which is {:.2f} % of the data"
#           .format(inconsistancy_counter,len(tbl),endState,startState,howOften))
    if len(incosistancy_tbl)>0:
        incosistancy_tbl.columns=tbl.index.names
        incosistancy_tbl.set_index(['Player','Date'],inplace=True)

    return(incosistancy_tbl,inconsistancy_counter)


# In[419]:


# delete_rows_with_inconsistency
def delete_inconsistancy_state(flag_inconsistancy,state, startState, endState):
    print(state.index.names)
    while flag_inconsistancy:
        
        incosistancy_tbl,inconsistancy_c=howOften_ends_comesAfter_Start(state,startState,endState)
        print('The number of inconsistancy for {} and {} is {}'.format(startState,endState,inconsistancy_c))
        if inconsistancy_c==0:
            
            print("---------------------------------------")
            print("The inconsistance is zero for {} and {}".format(startState,endState))
            print("---------------------------------------")
            
            flag_inconsistancy=False
            break
        counter=0
        for i,_ in tqdm(incosistancy_tbl.iterrows()):
            if i in state.index:
                state.drop(index=i,inplace=True, axis=0)
                counter +=1
        
        
    return(state)
    


# In[420]:


# adding a column named windowdiff that shows the day differnces between each events.  
def calculate_diff_days(tbl,tbl_with_windiff): 

    # pass each table for each person
    # remove the first row of the table
    # group the index by each person, so each time the table for each person will pass to the func
    for index, person_tbl in tqdm(tbl.groupby(level=0)): 
        if(len(person_tbl)>1):
            t1 = person_tbl[1:].iterrows() 

            #remove the last row of the table 
            t2 = person_tbl[:-1].iterrows() 
            # zip tow tables, so we can iterate (row1,row2), then (row2, row3)
            for (row1_index, row1), (row2_index, row2) in zip(t2, t1): 
                # change the windowdiff in the main tbl
                tbl_with_windiff.loc[row1_index].windowDiff=row2_index[1]-row1_index[1] 
                # keep the last row, so that we want this as -1 as this is the end of events for a person. 
                last_index=row2_index 
                # change the windowdiff for the last winsow to -1
            tbl_with_windiff.loc[last_index].windowDiff=-timedelta(days=1) 
    return(tbl_with_windiff)


# In[421]:



# preprocess the data(join state and event tables, fill the null values with zero, add windowdiff column)
def pre_processing(data_event_loc=data_event_loc,data_state_loc=data_state_loc):
    events = pd.read_hdf(data_event_loc)
    state = pd.read_hdf(data_state_loc)
    
    # Like event table we need player and date to be the index (multi index), so that we can merge them easily
    # 1- Reset index.
    # 2- Then drop the level_1 column which is unneccessary.
    # 3- Finally set the index as the player and Date.
    
    state.reset_index(inplace=True)
    state.drop(['level_1'],axis=1,inplace=True)
    state.set_index(['Player','Date'],inplace=True)
    
    # Delete the inconsistancy in the state dataset 
    flag_inconsistancy=True
    starts=['StartHospital','StartIcu','StartAntibiotics']
    ends=['EndHospital','EndIcu','EndAntibiotics']
    
    for row in zip(starts,ends):
        print("---------------------------------------")
        print("start deleting inconsistancy for {} and {}".format(row[0], row[1]))
        print("---------------------------------------")
        state=delete_inconsistancy_state(flag_inconsistancy,state, row[0], row[1])
    
    # get dummies for both tables.  
    state=pd.get_dummies(state['Tag'])
    events=pd.get_dummies(events['Tag'])
    
    # Build the main Table from merging both tables, event and state on the multi index.
    # As we perform outer merge, there will be some Nan value that has been fill by 0.
    # then the value of the table will be sort by player and time.   
    
    tbl=pd.merge(events,state, how='outer', on=['Player', 'Date'])
    tbl.fillna(0, inplace=True)
    tbl=tbl.sort_values(by=['Player','Date'])

    # We need a table with the window difference between the date and the next date.
    
    tbl_with_windiff=copy.deepcopy(tbl)
    tbl_with_windiff['windowDiff']=-timedelta(days=5)
    
    tbl_with_windiff=calculate_diff_days(tbl,tbl_with_windiff)
    
    print("---------------------------------------")
    print("Description of the window diff: ")
    print("---------------------------------------")
    print (tbl_with_windiff.windowDiff.describe())
    print("---------------------------------------")
    print("Events with more than 60 days differnece to the next event: ")
    print("---------------------------------------")
    print(tbl_with_windiff[tbl_with_windiff.windowDiff>timedelta(days=60)].windowDiff.describe())
    print("---------------------------------------")
    print("Palyers with only one record which should be removed while sessionazing:")
    print("---------------------------------------")
    print(tbl_with_windiff[tbl_with_windiff.windowDiff==timedelta(days=-5)].windowDiff.describe())
    print("---------------------------------------")
    return(tbl_with_windiff, state, events)


# In[422]:


def train_test_split_tbl(tbl,persons,person_tbl_train,person_tbl_test):
    # copy the main table into both train and test tables
    tbl_train=copy.deepcopy(tbl)
    tbl_test=copy.deepcopy(tbl)
    # we iterate through all the population That we have 
    for person in persons.iterrows():
        # if the person is in the train table ( that keep only persons), 
        # the person will be dropped from the test table
        
        if person_tbl_train.isin([person[1][0]]).any().any():
            tbl_test.drop(index=person[1][0], level=0,inplace=True, axis=0)
       
        # if the person is in the test table ( that keep only persons), 
        # the person will be dropped from the train table
        else: 
            tbl_train.drop(index=person[1][0], level=0,inplace=True, axis=0)
            
            
    # i keeps the value for the test length to check at the end that the perosns in the test_person are the same 
    # as the person in the test table
    i=0
    # it will iterate through the person_test table 
    for p in person_tbl_test.iterrows():
        # will check if the person in the person_test is in the test table. if yes, increament the i value.
        if pd.DataFrame(tbl_test.index.get_level_values(0).unique()).isin([p[1][0]]).any().any():
            i=i+1
            
    # after checking all the persons in the test table, we will check that if all the persons 
    # that are included in the test table are the ones that we are expecting to have
    
    if i==len(pd.DataFrame(tbl_test.index.get_level_values(0).unique())):
        print("---------------------------------------")
        print("Values have been checked and Train and Test split has been done correctly")
        print("---------------------------------------")
    return(tbl_train,tbl_test)


# In[423]:


# Creating Session column with the value of 0 at the begining.
# Then itrating through the rows and give an specific sessionId to the rows with the event duration 
# less than 60 days.
# Each iteration is based on the index which is player and the date. 
# So, if we have two events at the same date for one player, both of them will recieve the same session. 
# For example, in the data we have two events, including EndHospitl and End Antibiatic at the same date. 
# So both of them will recieve the same sessionId.

# if the event happened more the max of windowsize, cut the session and create another session for the next event
def Add_session_defined_by_windowsize(tbl, max_windowsize,min_windowsize): 
    tbl['sessionId']=0

    sessionId=1
    Flag=0
    for index, row in tqdm(tbl.iterrows()):


        if (row.windowDiff<timedelta(days=max_windowsize))&(row.windowDiff>timedelta(days=min_windowsize))&(Flag==0):
    #         print("First if ", index, sessionId)
            tbl.loc[index,'sessionId']=sessionId
            Flag=1
            continue;

        if (row.windowDiff<timedelta(days=max_windowsize))&(row.windowDiff>timedelta(days=min_windowsize))&(Flag==1):
    #         print("Second if ",index, sessionId)
            tbl.loc[index,'sessionId']=sessionId
            continue;

        if (row.windowDiff>timedelta(days=max_windowsize))&(Flag==1):
    #         print("Third if ",index, sessionId)
            tbl.loc[index,'sessionId']=sessionId
            sessionId+=1
            Flag=0
    len_=len(tbl[tbl['sessionId']==0])
    print("{} of the table recieved sessionId=0 which will be removed while sessionizing.". format(len_))
    print("---------------------------------------")
    return(tbl)


# In[424]:


# Event Table
def build_event_tbl(tbl):

    event_tbl=pd.DataFrame()
    # insert all events and delete the last 2 columns (sessionId and windowdiff)
    event_tbl['event']=tbl.columns[:-2]
    len_event=len(event_tbl['event'])
    event_tbl['eventId']=[i for i in range (1,len_event+1)]
    event_tbl.reset_index(drop=True, inplace=True)
    return(event_tbl)

# Session Table
def build_session_table(tbl):

    # defining a template table to keep the data and at end concate it to the session table 
    temp_session_tbl_prop=pd.DataFrame()
    session_tbl_prop=pd.DataFrame()
    temp_session_tbl_prop['sessionId']=[1]
    temp_session_tbl_prop['patient']=[1]
    temp_session_tbl_prop['startDate']=[1]
    temp_session_tbl_prop['EndDate']=[1]

    # as sessionIds with the value of zero do not belong to any epoch, these events need to be deleted.
    sessionId_list=np.delete(tbl['sessionId'].unique(),0)
    # for each session keep the start date, end date, and the name of the person. 
    for i in tqdm(sessionId_list): 
        temp_tbl=tbl[tbl['sessionId']==i]
        Flag_startDate=True
        for index,row in temp_tbl.iterrows():
            if Flag_startDate:
                Start_date=index[1].date()
                sessionId=row.sessionId
                patient=index[0]
                Flag_startDate=False

            End_date=index[1].date()
        # add them to the temp table and at the end concat the row which is built in the temp table 
        # to the main session table. 

        temp_session_tbl_prop['sessionId'].iloc[0]=sessionId
        temp_session_tbl_prop['patient'].iloc[0]=patient
        temp_session_tbl_prop['startDate'].iloc[0]=Start_date
        temp_session_tbl_prop['EndDate'].iloc[0]=End_date


        session_tbl_prop=pd.concat([session_tbl_prop,temp_session_tbl_prop])

    return(session_tbl_prop)

# Session_Event Table    
def build_session_event_table(tbl,event_tbl):
    # defining the session event table and a temp table to keep the data and then add it to the main table
    session_event_tbl=pd.DataFrame()
    temp_session_event_tbl=pd.DataFrame()
    temp_session_event_tbl['sessionId']=[1]
    temp_session_event_tbl['event']=[1]
    temp_session_event_tbl['index']=[1]
    
    idx=0
    sessionId_list=np.delete(tbl['sessionId'].unique(),0)
    for i in tqdm(sessionId_list): 
        # grab the part of the table with the same sessionId
        temp_tbl=tbl[tbl['sessionId']==i]
        temp_tbl.drop(['sessionId','windowDiff'], axis=1, inplace=True)
        # Iterate through the temp table
        for index,row in temp_tbl.iterrows():
                
                for item in zip(row.index,row):
                    # for each row if there is an event which is one, it will be added as a separate row to the table
                    if item[1]:
#                         event=event_tbl[event_tbl.event==item[0]].iloc[0].event
                        event=event_tbl[event_tbl.event==item[0]].iloc[0].eventId
                        temp_session_event_tbl['event']=event
                        temp_session_event_tbl['sessionId']=i
                        temp_session_event_tbl['index']=idx
                        idx=idx+1
                        session_event_tbl=pd.concat([session_event_tbl,temp_session_event_tbl])

    return(session_event_tbl.set_index('index'))    


# In[425]:


# we want the sessions with the least length of 3, so the new tbl will contain all the session with the lenght of >3
def remove_session_with_len_lessthan_three(session_event_tbl):
    session_event_tbl_new=pd.DataFrame(columns=['sessionId','eventid','index'])
    First_row_Flag=True
    Flag_correct=True
    for index,tbl in tqdm(session_event_tbl.groupby(session_event_tbl['sessionId'])):
        if len(tbl)>2:
            if First_row_Flag:
                session_event_tbl_new=tbl
                First_row_Flag=False
            else:
                session_event_tbl_new=session_event_tbl_new.append(tbl)
                
    for index,tbl in session_event_tbl_new.groupby(session_event_tbl_new['sessionId']):
        if len(tbl)<2:
            Flag_correct=False
    if Flag_correct:
        print("All the sessions with the lenght of < 2 have been removed correctly = {}"
              .format(len(session_event_tbl)-len(session_event_tbl_new)))
        print("-------------------------------------------------------------------")
#     return(session_event_tbl_new.set_index('index'))
    return(session_event_tbl_new)

# We need to turn the session_event_tbl_new to a sereis of sessions
def creat_session_sequence(df):
    sequence_session=[]
    group_by_session=df.groupby(df['sessionId'])
    for i,group in tqdm(group_by_session):
        sequence_session.append(list(group['event'].values))
    return(sequence_session)


# In[426]:


#https://github.com/fastforwardlabs/session_based_recommenders/blob/c438dd1334fcefc6bedea69b0cd67f779a5de5d3/recsys/data.py#L76
def x_y_split(session_sequences):
    """
    In Next Event Prediction (NEP), training is perform on the first n-1 items in a session sequence of n items. 
    The target set is constructed of nth item 
    Example:
        Given a session sequence ['045', '334', '342', '8970', '128']
        Training is done on ['045', '334', '342', '8970']
        target (and validation) is done on ['128']
    """
    np.random.seed(123)
    
    ### Construct training set
    # use (1 st, ..., n-1 th) items from each session sequence to form the train set (drop last item)
    
    x = [sess[:-1] for sess in session_sequences]
    ### Construct targer by seperating the nth item from the session_sequence and validation sets
    y = [sess[-1:] for sess in session_sequences]
    return x, y


# In[427]:


# adding zero padding to the end of the training sequence of the data to reach 
# to the max length of the seq that we have 
def padding_the_sequence(train_sequence_session,test_sequence_session, pading_num):
    
    max_sentence_length=0
    for l in train_sequence_session:
        if len(l)>max_sentence_length:
            max_sentence_length=len(l)
    
    max_sentence_length_test=0
    for l in test_sequence_session:
        if len(l)>max_sentence_length_test:
            max_sentence_length_test=len(l)
            
    if max_sentence_length_test>max_sentence_length:
        max_sentence_length=max_sentence_length_test
        
    # add the padding at the end of each sentence, so the lengh of each one will be the same. 
    s_train=(list(map(lambda x: pd.np.pad(x, (0, max_sentence_length-len(x)),'constant', constant_values=(pading_num)), train_sequence_session)))
    s_test=(list(map(lambda x: pd.np.pad(x, (0, max_sentence_length-len(x)),'constant', constant_values=(pading_num)), test_sequence_session)))

    # pytorch convert from numpy arrays to tensors.
    list_of_ternsor_train=list(map(lambda x: torch.LongTensor(x),s_train))
    list_of_ternsor_test=list(map(lambda x: torch.LongTensor(x),s_test))
    
    #stack the list of tensor and return back the lists as a tensor 
    t_train=torch.stack(list_of_ternsor_train)
    t_test=torch.stack(list_of_ternsor_test)
    print("max_sentence_length is : {}, the shape_train is {}, the shape_test is {}"
          .format(max_sentence_length,t_train.shape, t_test.shape))
    return(t_train,t_test,max_sentence_length)



