# About Experiments
To make it easier to submit jobs to the cluster via `sbatch`, I've created a separate batch script for each model training experiment. 
These are all basically the same code, but the model parameters, data seeds and output paths vary.

The `multipa-evaluate` command may be used to compare multiple trained models on the test set after training. 

Experiments corresponding with script name prefixes are described as follows:

## `hyperparam_tuning`
Vary model parameters like learning rates and batch size (using the same training data in each experiment) to establish a reasonable baseline.
Note: effective batch size = batch per device x gradient accumulation steps x num GPUs

Goals:
- Figure out which model parameters produce good performance
- Establish baseline for our model architecture on the Buckeye corpus
- Check for any warning signs that the model architecture may not be appropriate, like over/underfitting


Params to vary: 
- Effective batch size: [64, 32] (achieve these by varying batch size per device, number of gpus and grad accumulation steps appropriately)
    - To complete training quickly, you can use 4 or 8 GPUs on Unity, but they have to be large to get enough VRAM. 
    - Note: effective batch size = batch per device x gradient accumulation steps x num GPUs
- Learning rate [3e-4, 3e-5, 9e4]


## `data_seed` 
Vary the random seed to select training data while keeping an even 50/50 gender split to measure statistical significance of changing training data selection. Retrain with the same model parameters, but different data seeding to measure statistical significance of data seed, keeping 50/50 gender split. 

Goals: 
- Establish whether data variation with the same gender makeup is statistically significant in changing performance on the test set

Params to vary:
- training data seed (--train_seed): [7 (default), 91, 15, 139, 503]


## `gender_split`
Still training with a total amount of data equal to half the full training data (4000 examples), vary the gender split 30/70, but still drawing examples from all individuals. Do 5 models for each gender split with the same model parameters but different data seeds. 

Goals: 
- Determine how different in gender split in training data affects performance

Params to vary: 
- percent female (--percent_female) [0.3, 0.7]
- training seed (--train_seed): [359, 130, 809, 700, 114]


## `vary_individuals`
