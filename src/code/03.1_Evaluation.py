#!/usr/bin/env python
# coding: utf-8

# # 3. Performance Metrics
# 
# Take trained models from **02_BuildModelsKGE.ipynb** and check their performance metrics.
# 
# We want evaluation scores for:
# * the whole dataset of villageB
# * event data of villageB
# * inferred data of villageB
# * explanation data of villageB

# In[2]:


from helper import *


# ## 1. load models

# In[4]:


# TransE_Adam_default_1000E
TransE_default_model = load_model("TransE_Adam_default_100E")
# TransE_Adam_HPO_1000E
TransE_HPO_model = load_model("TransE_Adam_hpo_100EbestModel")
# ComplEx_Adam_default_1000E
ComplEx_default_model = load_model("ComplEx_Adam_default_100E")
# ComplEx_Adam_hpo_1000E
ComplEx_HPO_model = load_model("ComplEx_Adam_hpo_100EbestModel")
# TransH_Adam_default_1000E
TransH_default_model = load_model("TransH_Adam_default_1000E")
# TransH_Adam_default_100E
TransH_default_model = load_model("TransH_Adam_default_100E")
# TransH_Adam_HPO_1000E
TransH_HPO_model = load_model("TransH_Adam_hpo_100EbestModel")


models = {
    "TransE_default" : TransE_default_model,
    "TransE_hpo" : TransE_HPO_model,
    "ComplEx_default" : ComplEx_default_model,
    "ComplEx_hpo" : ComplEx_HPO_model,
    "TransH_default" : TransH_default_model,
    "TransH_hpo" : TransH_HPO_model,
    }


# ## 2. load data

# In[6]:


event_triples_B = get_triples("../data/graphs/event_data_B.csv")
explanation_triples_B = get_triples("../data/graphs/explanation_data_B.csv")
inferred_triples_B = get_triples("../data/graphs/inferred_data_B.csv")
time_triples_B = get_triples("../data/graphs/time_data_B.csv")
topology_triples_B = get_triples("../data/graphs/topology_data_B.csv")
all_triples_B = get_triples("../data/graphs/all_data_B.csv")

all_triples_A  = get_triples("../data/graphs/all_data_A.csv")

evaluation_triples = {
    "all" : all_triples_B,
    "events" : event_triples_B,
    "explanation" : explanation_triples_B,
    "inferred" : inferred_triples_B
    }


# ## 3. evaluate models on data

# In[8]:


for model in models:
    for triples in evaluation_triples:
        evaluate_model_on(
            models.get(model), 
            evaluation_triples.get(triples), 
            all_triples_A,
            model + '_' + triples)
        print("Evaluation saved as: " + model + "_" + triples)

