# About Experiments
To make it easier to submit jobs to the cluster via `sbatch`, I've created a separate batch script for each model training experiment. 
These are all basically the same code, but the model parameters, data seeds and output paths vary.

The `multipa-evaluate` command may be used to compare multiple trained models on the test set after training. 

Experiments corresponding with script names are described as follows:

## `hyperparam_tuning`
Vary model parameters like learning rates and batch size (using the same training data in each experiment) to establish a reasonable baseline.
Note: effective batch size = batch per device x gradient accumulation steps x num GPUs

## `data_seed` 
Vary the random seed to select training data while keeping an even 50/50 gender split to measure statistical significance of changing training data selection.

## `gender_split`


## `vary_individuals`
