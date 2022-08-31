#!/usr/bin/env python
# coding: utf-8

# # 2. Build Models
# 
# Use preprocessed data from notebook **01_Preprocessing.ipynb** and train KGE models on the data.

# In[ ]:


#! /usr/venv/thesis/bin/python3


# In[1]:


import sys
root_path = '..'
sys.path.append(root_path)
get_ipython().run_line_magic('pip', 'install -r ../requirements.txt')


# In[2]:


from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import sparql_dataframe
import pandas as pd
import torch
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import pickle


# ## Import Data

# In[3]:


with open('../data/splits/entity_to_id.pkl', 'rb') as f:
    entity_to_id = pickle.load(f)

with open('../data/splits/relation_to_id.pkl', 'rb') as f:
    relation_to_id = pickle.load(f) 
    
def get_triples(fileName):
    df = pd.read_pickle(fileName)
    triples = TriplesFactory.from_labeled_triples(df[["s", "p", "o"]].values,
                                              entity_to_id=entity_to_id,
                                              relation_to_id=relation_to_id,)
    return triples


# In[4]:


training = get_triples('../data/splits/training.pkl')
testing = get_triples('../data/splits/test.pkl')
validation = get_triples('../data/splits/validation.pkl')


# ## Building Embedding Pipeline with default values, no HPO
# 
# try to put transE with Adam optimizer in a pipeline  - no hpo for now
# 
# ### TransE, Adam Optimizer

# In[ ]:


# TransE, Adam Optimizer, no HPO
MODEL_NAME = 'TransE'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 1000
result = pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
        checkpoint_name = MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_'+str(NUM_EPOCHS)+'E.pt',
        checkpoint_directory = "../checkpoints/",
        checkpoint_frequency = 30,
    ),
    stopper = 'early',
    random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E')
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E' + ".csv")

print("-------------------\n Successfully trained TransE_Adam_default_1000E \n-------------------")


# ### ComplEx, Adam Optimizer

# In[ ]:


# ComplEx, Adam Optimizer, no HPO
MODEL_NAME = 'ComplEx'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 1000
result = pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
        checkpoint_name = MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_'+str(NUM_EPOCHS)+'E.pt',
        checkpoint_directory = "../checkpoints/",
        checkpoint_frequency = 30,
    ),
    stopper = 'early',
    random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E')
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E' + ".csv")

print("-------------------\n Successfully trained ComplEx_Adam_default_1000E \n-------------------")


# ## Use HPO for Model Training
# 
# ### TransE, Adam

# In[ ]:


from pykeen.hpo import hpo_pipeline

# Transe, Adam Optimizer, HPO
MODEL_NAME = 'TransE'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 1000
result = hpo_pipeline(
    n_trials = 30,
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
    ),
    save_model_directory="../checkpoints/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_'+str(NUM_EPOCHS)+'E/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E')

print("-------------------\n Successfully trained TransE_Adam_HPO_1000E \n-------------------")


# In[26]:


import json
file = '../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E/best_pipeline/pipeline_config.json'

with open(file) as json_file:
    config = json.load(json_file)
    
config["pipeline"]["training"] = training
config["pipeline"]["testing"] = testing
config["pipeline"]["validation"] = validation


# In[29]:


from pykeen.pipeline import pipeline_from_config

result = pipeline_from_config(config)
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + "bestModel")
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + "bestModel" + ".csv")

print("-------------------\n Successfully evaluated TransE_Adam_hpo_1000E \n-------------------")


# ### ComplEx, Adam

# In[ ]:


# ComplEx, Adam Optimizer, HPO
MODEL_NAME = 'ComplEx'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 1000
result = hpo_pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
    ),
    save_model_directory="../checkpoints/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_'+str(NUM_EPOCHS)+'E/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory(root_path +'/models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E')
model = result.model
metrics = result.metric_results.to_df()
metrics.to_csv(root_path + "/model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + ".csv")

print("-------------------\n Successfully trained ComplEx_Adam_HPO_1000E \n-------------------")


# In[ ]:


import json
file = '../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E/best_pipeline/pipeline_config.json'

with open(file) as json_file:
    config = json.load(json_file)
    
config["pipeline"]["training"] = training
config["pipeline"]["testing"] = testing
config["pipeline"]["validation"] = validation


# In[ ]:


from pykeen.pipeline import pipeline_from_config

result = pipeline_from_config(config)
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + "bestModel")
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + "bestModel" + ".csv")

print("-------------------\n Successfully evaluated ComplEx_Adam_hpo_1000E \n-------------------")

