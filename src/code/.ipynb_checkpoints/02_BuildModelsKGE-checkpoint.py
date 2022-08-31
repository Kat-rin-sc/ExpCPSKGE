#!/usr/bin/env python
# coding: utf-8

# # 2. Build Models
# 
# Use preprocessed data from notebook **01_Preprocessing.ipynb** and train KGE models on the data.

# In[1]:


#! /usr/venv/thesis/bin/python3


# In[2]:


import sys
root_path = '..'
sys.path.append(root_path)
get_ipython().system('pip install -r ../requirements.txt ')


# In[3]:


from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import sparql_dataframe
import pandas as pd
import torch
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split


# ## Import Data

# In[7]:


training_data = pd.read_pickle('../data/splits/training.pkl')
test_data = pd.read_pickle('../data/splits/test.pkl')
validation_data = pd.read_pickle('../data/splits/validation.pkl')
all_data = training_data.append(test_data).append(validation_data)

all_triples = TriplesFactory.from_labeled_triples(all_data[["s", "p", "o"]].values)

training = TriplesFactory.from_labeled_triples(training_data[["s", "p", "o"]].values,
                                              entity_to_id=all_triples.entity_to_id,
                                              relation_to_id=all_triples.relation_to_id,)
testing = TriplesFactory.from_labeled_triples(test_data[["s", "p", "o"]].values,
                                              entity_to_id=all_triples.entity_to_id,
                                              relation_to_id=all_triples.relation_to_id,)
validation = TriplesFactory.from_labeled_triples(validation_data[["s", "p", "o"]].values,
                                              entity_to_id=all_triples.entity_to_id,
                                              relation_to_id=all_triples.relation_to_id,)


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
model = result.model
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E' + ".csv")
print("-------------------\n Successfully trained TransE_Adam_default_1000E \n-------------------"


# In[ ]:


# import torch
# from pykeen.models import predict

# model = torch.load('../models/TransE_Adam_default_1000E/trained_model.pkl')

# # Score top K triples
# top_k_predictions_df = predict.get_all_prediction_df(model, k=150, triples_factory=validation)
# top_k_predictions_df.to_pickle("../predictions/top_150_overall_TransE_Adam_default_1000E.pkl")

# # get predictions for inferred


# print("-------------------\n Successfully predicted on TransE_Adam_default_1000E \n-------------------"


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
model = result.model
result.plot_losses()
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_default_' + str(NUM_EPOCHS) + 'E' + ".csv")
print("-------------------\n Successfully trained ComplEx_Adam_default_1000E \n-------------------"


# In[ ]:





# ## Use HPO for Model Training
# 
# ### TransE, Adam
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
    save_model_directory="/checkpoints/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_'+str(NUM_EPOCHS)+'E/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E')
model = result.model
result.plot_losses()
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_' + OPTIMIZER_NAME + '_hpo_' + str(NUM_EPOCHS) + 'E' + ".csv")

print("-------------------\n Successfully trained TransE_Adam_HPO_1000E \n-------------------"
# In[ ]:





# In[ ]:





# In[ ]:




