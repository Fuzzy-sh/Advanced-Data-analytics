#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=5:30:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

#SBATCH --job-name=main_program
#SBATCH --output=main_program_%A_%a.out
                
## Main program, dividing the data based on the parameter and train the model 

echo starting slurm script to run the main program to split the test and train, then train the model and save the results for each params

date
id

echo start initialization


which python
conda env list

echo finished preprocessing 

echo read in the inputes:

index=0
while read line ; do
        LINEARRAY[$index]="$line"
        index=$(($index+1))
done < params.txt

echo $((${SLURM_ARRAY_TASK_ID}-1))
echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting main program python code

echo python main_program.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 
python main_program.py  ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo ending slurm script to do main program

date
