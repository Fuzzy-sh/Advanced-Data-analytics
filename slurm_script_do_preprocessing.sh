#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=5:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=Preprocessing 
#SBATCH --output=preprocessing_%j.out

## Pre_processing : in which, I am willing to create three tables for session, event, and the connector table which is session_event

echo starting slurm script to do preprocessing 
# to print our some infor for the start, like runtime, date and id of the job
#runtime
date
id

echo start initialization
# As if we dont activate the talc environment before running the codes, we need the following coding to know the activate the environment first, otherwise there will be errors.


which python
source env list

echo finished initializaing

# perform the cleaning and data preprocessing
echo starting python code
echo main_preprocess_clean_data.py
python main_preprocess_clean_data.py

echo ending slurm script to perform preprocessing
