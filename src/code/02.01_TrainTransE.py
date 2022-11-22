#!/usr/bin/env python
# coding: utf-8

# # 2. Build Models
# 
# Use preprocessed data from notebook **01_Preprocessing.ipynb** and train KGE models on the data.

# In[ ]:


#! /usr/venv/thesis/bin/python3


# In[3]:


# %pip install -r ../requirements.txt 


# In[1]:


from helper import *


# ## Import Data

# In[2]:


training = get_triples('../data/splits/trainingNT.pkl')
testing = get_triples('../data/splits/testNT.pkl')
validation = get_triples('../data/splits/test_validNT.pkl')


# ## Building Embedding Pipeline with default values, no HPO
# 
# try to put transE with Adam optimizer in a pipeline  - no hpo for now
# 
# ### TransE, Adam Optimizer

# In[3]:


# TransE, Adam Optimizer, no HPO
MODEL_NAME = 'TransE'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 100


# In[ ]:


result = pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
        checkpoint_name = MODEL_NAME +'_default.pt',
        checkpoint_directory = "../checkpoints/",
        checkpoint_frequency = 30,
    ),
    stopper = 'early',
    random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME +'_default')
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME +'_default.csv')

print("-------------------\n Successfully trained " + MODEL_NAME  + '_default' + "\n-------------------")

torch.cuda.empty_cache() 


# ## Use HPO for Model Training
# 
# ### TransE, Adam

# In[4]:


from pykeen.hpo import hpo_pipeline
from optuna.samplers import GridSampler


# Transe, Adam Optimizer, HPO
result = hpo_pipeline(
    n_trials = 30,
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,   
    sampler=GridSampler,
    sampler_kwargs=dict(
        search_space={
            "model.embedding_dim": [32, 64, 128],
            "model.scoring_fct_norm": [2],
            "loss": 'marginranking',
            "loss.margin": [1.0,2.0],
            "optimizer.lr": [1.0e-03],
            "negative_sampler":'basic',
            "negative_sampler.num_negs_per_pos": [32],
            "training.num_epochs": [100],
            "training.batch_size": [128],
            "stopper":'early',
            "optimizer":["Adam", "SGD"]
            
        },
    ),
#    optimizer = OPTIMIZER_NAME,
    save_model_directory="../checkpoints/" + MODEL_NAME +'_hpo/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME +'_hpo')

print("-------------------\n Successfully trained" + MODEL_NAME + '_hpo' + "\n-------------------")

torch.cuda.empty_cache() 


# In[26]:


import json
file = '../models/' + MODEL_NAME + '_hpo' + '/best_pipeline/pipeline_config.json'

with open(file) as json_file:
    config = json.load(json_file)
    
config["pipeline"]["training"] = training
config["pipeline"]["testing"] = testing
config["pipeline"]["validation"] = validation


# In[29]:


from pykeen.pipeline import pipeline_from_config

result = pipeline_from_config(config)
result.save_to_directory('../models/' + MODEL_NAME + '_hpo' + "bestModel")
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME +'_hpo' + "bestModel" + ".csv")

print("-------------------\n Successfully evaluated " + MODEL_NAME + '_hpo' + "\n-------------------")

torch.cuda.empty_cache() 

