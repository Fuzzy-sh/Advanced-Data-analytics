#!/bin/bash

echo starting the top level Fuzzy script
window_size_max=( 60 50 40 30 )
train_test_split_rate=( 0.90 0.80 )
batch_size=( 16 32 64 )
lr=( 0.001 0.0001 )
num_neurons_per_layer=( 16 32 64 128 256 )
num_hiddin_layer=( 1 2 3 )
mode=( 'mean' 'max' 'sum' )

job1Counter=0
echo $job1Counter

for window_size_max in "${window_size_max[@]}" ; do
for train_test_split_rate in "${train_test_split_rate[@]}" ; do
for batch_size in "${batch_size[@]}" ; do
for lr in "${lr[@]}" ; do 
for num_neurons_per_layer in "${num_neurons_per_layer[@]}" ; do
for num_hiddin_layer in "${num_hiddin_layer[@]}" ; do
for mode in "${mode[@]}" ; do
	echo ${window_size_max} ${train_test_split_rate} ${batch_size} ${lr} ${num_neurons_per_layer} ${num_hiddin_layer} ${mode} >> params.txt
	job1Counter=$((job1Counter+1))
done
done
done
done
done
done
done 


echo $job1Counter
# First, do preprocessing

echo Starting Preprocessing from top level Fuzzy script
# print out the date and time for now. 
date

# Run the first .sh script which is welcoming and keep the jobId. Then print the jobId for this script. 

jobID1=$(sbatch slurm_script_do_preprocessing.sh)
echo $jobID1

# Second, Start main_program
# define how many job array we want to run. Then print out the sbaching which is dependend on the previous jobId.

echo sbatch  --dependency=afterok:${jobID1##* } --array=0-$job1Counter slurm_script_do_main_program.sh

# Run the second job after the first job has been performed ok. The main job will be performed in several job arrays.
jobID2=$(sbatch --dependency=afterok:${jobID1##* } --array=0-$job1Counter slurm_script_do_main_program.sh)
echo $jobID2

