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
Still training with a total amount of data equal to half the full training data (4000 examples), vary the gender split 30/70, but draw examples from all individuals. Do 5 models for each gender split with the same model parameters but different data seeds. 

Goals: 
- Determine how different in gender split in training data affects performance

Params to vary: 
- percent female (--percent_female) [0.3, 0.7]
- training seed (--train_seed): [359, 130, 809, 700, 114]


## `vary_individuals`
Still keeping the total amount of data equal to half the training data and the gender split 50/50, exclude certain speakers completely. Train with the same model parameters but different individuals each time. 
For reference, the speakers and their demographics included in the training data are: 

| speaker_id | speaker_gender | speaker_age_range | 
| ---------- | -------------- | ----------------- |
| S01 | f | y |
| S04 | f | y | 
| S08 | f | y | 
| S09 | f | y | 
| S12 | f | y | 
| S21 | f | y | 
| S02 | f | o |
| S05 | f | o | 
| S07 | f | o | 
| S14 | f | o | 
| S16 | f | o |
| S17 | f | o | 
| S06 | m | y | 
| S11 | m | y | 
| S13 | m | y | 
| S15 | m | y | 
| S28 | m | y | 
| S30 | m | y |
| S03 | m | o | 
| S10 | m | o | 
| S19 | m | o |
| S22 | m | o |
| S24 | m | o | 


Goals: 
- Determine how variety of speakers in the training data affects performance

Params to vary: 
- demographic make up of training data by age, using --speaker_restriction 
    - Experiments 1-3: only "young" individuals
    - Experiments 4-6: only "old" individuals