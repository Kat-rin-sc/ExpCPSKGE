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

# ### ComplEx, Adam Optimizer

# In[ ]:


# ComplEx, Adam Optimizer, no HPO
MODEL_NAME = 'RotatE'
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 1000


# In[ ]:


result = pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        batch_size=1,
        num_epochs = NUM_EPOCHS,
        checkpoint_name = MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_'+str(NUM_EPOCHS)+'E.pt',
        checkpoint_directory = "../checkpoints/",
        checkpoint_frequency = 30,
    ),
    evaluation_kwargs=dict(batch_size=16),
    stopper = 'early',
    random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E')
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E' + ".csv")

print("-------------------\n Successfully trained " + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E'+ "\n-------------------")


# ## Use HPO for Model Training

# ### ComplEx, Adam

# In[ ]:


# ComplEx, Adam Optimizer, HPO
result = hpo_pipeline(
    n_trials = 30,
    training = training,
    validation = validation,
    testing = testing,
    model = MODEL_NAME,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        batch_size=1,
        num_epochs = NUM_EPOCHS,
    ),
    evaluation_kwargs=dict(batch_size=1),
    save_model_directory="../checkpoints/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_'+str(NUM_EPOCHS)+'E/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E')

print("-------------------\n Successfully trained " + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E'+ " \n-------------------")


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

print("-------------------\n Successfully evaluated " + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E'+ "\n-------------------")

