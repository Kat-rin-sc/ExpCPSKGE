#!/usr/bin/env python
# coding: utf-8

# # 2. Build Models
# 
# ## Run TTransE with TKGE Framework
# 
# ### Load Libraries and Data

# In[1]:


import pickle as pkl
import pandas as pd
import os


# Load data as pandas dataframes and store in the respective data folder of the framework

# In[ ]:


training = pd.read_pickle(r"../data/splits/trainingT.pkl")
training = training.loc[training.p != "expcps:refValues"]
training.to_csv("TKGE/data/ExpCPS/train.txt", index = False, sep = "\t", header = False)

test = pd.read_pickle(r"../data/splits/testT.pkl")
test = test.loc[test.p != "expcps:refValues"]
test.to_csv("TKGE/data/ExpCPS/test.txt", index = False, sep = "\t", header = False)

valid = pd.read_pickle(r"../data/splits/validT.pkl")
valid = valid.loc[valid.p != "expcps:refValues"]
valid.to_csv("TKGE/data/ExpCPS/valid.txt", index = False, sep = "\t", header = False)


# In[8]:


inferred = pd.read_csv("../data/graphs/inferred_data_B_quads.csv", index_col = 0)
inferred.to_csv("TKGE/data/ExpCPS/inferred.txt", index = False, sep = "\t", header = False)

explanation = pd.read_csv("../data/graphs/explanation_data_B_quads.csv", index_col = 0)
explanation.to_csv("TKGE/data/ExpCPS/explanation.txt", index = False, sep = "\t", header = False)

events = pd.read_csv("../data/graphs/event_data_B_quads.csv", index_col = 0)
events.to_csv("TKGE/data/ExpCPS/events.txt", index = False, sep = "\t", header = False)

all_data = pd.read_csv("../data/graphs/all_data_B_quads.csv", index_col = 0)
all_data.to_csv("TKGE/data/ExpCPS/all.txt", index = False, sep = "\t", header = False)


# In[ ]:


os.getcwd()
os.chdir("./TKGE/")
os.getcwd()


# In[ ]:


get_ipython().run_line_magic('run', 'tkge.py train --config "./ExpCPS_config/TTransE_ExpCPS.yaml"')


# In[ ]:


dir_list = os.listdir("./ExpCPS_checkpoints/TTransE_ExpCPS_experiment/ttranse/ExpCPS/")
dir_list.sort()
bestEx = dir_list[-1]
bestEx = ".\\ExpCPS_checkpoints\\TTransE_ExpCPS_experiment\\ttranse\\ExpCPS\\"+bestEx
bestEx


# In[ ]:


get_ipython().run_line_magic('run', 'tkge.py eval --config "./ExpCPS_config/TTransE_ExpCPS.yaml"')


# In[ ]:


get_ipython().run_line_magic('run', 'tkge.py eval --ex $bestEx')


# In[ ]:




