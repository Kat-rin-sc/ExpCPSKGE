#!/usr/bin/env python
# coding: utf-8

# # Code for Preprocessing of Knowledge Graph
# 
# Only run this script locally, where the right data is imported to GraphDB and GraphDB is running

# ## Setup and loading libraries

# In[1]:


import sys
root_path = '..'
sys.path.append(root_path)
import sparql_dataframe
import pandas as pd
import torch
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from helper import *


# ## load Data using SPARQL
# 
# ### Graph Names

# In[2]:


# define prefixes used in queries
prefixes = """
PREFIX expcps: <http://expcps.link/vocab#>
PREFIX wgs: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX test: <http://expcps.link/expcps/SoftProtection#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX fno: <https://w3id.org/function/ontology#>
PREFIX gn: <http://www.geonames.org/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX prop: <http://expcps.link/expcps/properties#>

"""

# define function to query data
def get_query_result(query, query_name):
    # repository source
    endpoint = "http://localhost:7200/repositories/test1"

    df = sparql_dataframe.get(endpoint, query, post = True)

    # replace namespaces by prefixes
    df = df.replace("http://expcps.link/vocab#", "expcps:", regex=True)
    df = df.replace("http://www.w3.org/2003/01/geo/wgs84_pos#", "wgs:", regex=True)
    df = df.replace("http://expcps.link/expcps/SoftProtection#", "test:", regex=True)
    df = df.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:", regex=True)
    df = df.replace("http://www.w3.org/2002/07/owl#", "owl:", regex=True)
    df = df.replace("https://w3id.org/function/ontology#", "fno:", regex=True)
    df = df.replace("http://www.geonames.org/ontology#", "gn:", regex=True)
    df = df.replace("http://www.w3.org/2001/XMLSchema#", "xsd:", regex=True)
    df = df.replace("http://www.w3.org/2000/01/rdf-schema#", "rdfs:", regex=True)
    df = df.replace("http://www.w3.org/ns/sosa/", "sosa:", regex=True)
    df = df.replace("http://expcps.link/expcps/properties#", "prop:", regex=True)


    # save to csv
    df.to_csv('../data/graphs/' + query_name +'.csv')
    return df

# get all named graphs in repository
graph_names_query = """
        SELECT DISTINCT ?graphName 
        WHERE {
        GRAPH ?graphName { ?s ?p ?o }
        }
        """

graph_names = get_query_result(graph_names_query, "graphNames")

graph_names.head()


# ### time_data - all timeseries data points
# #### Village A

# In[3]:


# get timeseries data
time_graphs = graph_names.loc[graph_names.graphName.str.startswith("http://expcps.link/update/")]

source = "FROM <" + time_graphs["graphName"].str.cat( sep = ">\nFROM <") + ">"
# build query
q = prefixes + "select *" + source + "where{?s ?p ?o}"

time_data = get_query_result(q, "time_data_A")
time_data.head()


# In[4]:


time_data.shape


# In[5]:


time_data.loc[time_data.p == "sosa:resultTime"].groupby("o").count()


# In[6]:


time_data = time_data.loc[~time_data.o.isin([
    "1970-01-01T00:00:00", 
    "1970-01-01T00:00:01",
    "1970-02-16T07:00:40",
    "1970-06-20T19:00:40",
    "1972-01-15T20:00:40",
    "1972-01-19T20:00:40",
    "1973-07-29T02:00:40",
    "1973-10-08T11:00:40",
    "1973-10-18T03:00:40",
    "1973-10-23T23:00:40",
    "1973-12-05T04:00:40",
    "1973-12-09T06:00:40"
])]
time_data.to_csv('../data/graphs/time_data_A.csv')


# In[7]:


time_data.shape


# In[8]:


time_data.loc[time_data.p == "sosa:resultTime"].groupby("o").count()


# #### Village B - unfinished

# In[9]:


# get timeseries data
time_graphs = graph_names.loc[graph_names.graphName.str.startswith("http://expcps.link/update/")]

source = "FROM <" + time_graphs["graphName"].str.cat( sep = ">\nFROM <") + ">"
# build query
q = prefixes + "select *" + source + "where{?s ?p ?o}"

time_data_B = get_query_result(q, "time_data_B")
time_data_B = time_data.loc[~time_data.o.isin([
    "1970-01-01T00:00:00", 
    "1970-01-01T00:00:01",
    "1970-02-16T07:00:40",
    "1970-06-20T19:00:40",
    "1972-01-15T20:00:40",
    "1972-01-19T20:00:40",
    "1973-07-29T02:00:40",
    "1973-10-08T11:00:40",
    "1973-10-18T03:00:40",
    "1973-10-23T23:00:40",
    "1973-12-05T04:00:40",
    "1973-12-09T06:00:40"
])]
time_data_B.to_csv('../data/graphs/time_data_B.csv')


# ### topology_data - setup of Knowledge Graph (Ontology)
# #### Village A

# In[10]:


# get topology data
topology_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/topology/SoftProtection"])]
source = "FROM <" + topology_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
topology_data = get_query_result(q, "topology_data_A")
topology_data.head()


# In[11]:


topology_data.shape


# #### Village B - unfinished

# In[12]:


# get topology data
topology_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/topology/SoftProtection"])]
source = "FROM <" + topology_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
topology_data_B = get_query_result(q, "topology_data_B")
topology_data_B.head()


# ### inferred__data - potential causalities between observations
# #### Village A

# In[13]:


# get inferred data from causality relations
inferred_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/inferred/SoftProtection"])]
source = "FROM <" + inferred_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
inferred_data = get_query_result(q, "inferred_data_A")
inferred_data.head()


# In[14]:


inferred_data.shape


# #### Village B - unfinished

# In[15]:


# get inferred data from causality relations
inferred_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/inferred/SoftProtection"])]
source = "FROM <" + inferred_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
inferred_data_B = get_query_result(q, "inferred_data_B")
inferred_data_B.head()


# ### events_data - all events based on some criteria
# #### Village A

# In[16]:


# get events data
event_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/event/SoftProtection"])]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
event_data = get_query_result(q, "event_data_A")
event_data.head()


# In[17]:


event_data.shape


# #### Village B - unfinished

# In[18]:


# get events data
event_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/event/SoftProtection"])]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
event_data_B = get_query_result(q, "event_data_B")
event_data_B.head()


# ### explanation_data - two possible paths of events
# #### Village A

# In[19]:


# get events data
event_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/explanation/SoftProtection"])]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
query1 = prefixes + "SELECT ?effect ?causedBy ?cause\n" + source + """
WHERE {?Argument expcps:causeEvent ?cause .
        ?Argument expcps:effectEvent ?effect}
"""
explanation_data = get_query_result(query1, "explanation_data_A")
explanation_data["causedBy"] = "expcps:causedBy"
explanation_data.columns = ["s", "p", "o"]
explanation_data.head()
explanation_data.to_csv('../data/graphs/explanation_data_A.csv')


# In[20]:


explanation_data.shape


# #### Village B - unfinished

# In[21]:


# get events data
event_input = graph_names.loc[graph_names.graphName.isin(["http://expcps.link/explanation/SoftProtection"])]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
query1 = prefixes + "SELECT ?effect ?causedBy ?cause\n" + source + """
WHERE {?Argument expcps:causeEvent ?cause .
        ?Argument expcps:effectEvent ?effect}
"""
explanation_data_B = get_query_result(query1, "explanation_data_B")
explanation_data_B["causedBy"] = "expcps:causedBy"
explanation_data_B.columns = ["s", "p", "o"]
explanation_data_B.head()
explanation_data_B.to_csv('../data/graphs/explanation_data_B.csv')


# ## Training, Test, Validation Splits
# 
# **Data available:**
# * **topology_data** - data on the topology of the graph, no timeseries data
# * **time_data** - all the time series data derived from BIFROST
# * **inferrred_data** - possible explanations inferred by the graph
# * **event_data** - detected events in the KG
# * **explanation_data** - all explanations for events based in the explanations graph, but reducing to caused_by relations

# In[22]:


from pykeen.triples import TriplesFactory
import pickle

all_data_A = time_data.append(topology_data).append(inferred_data).append(event_data).append(explanation_data)
all_data_A.to_csv("../data/graphs/all_data_A.csv")
all_data_B = time_data_B.append(topology_data_B).append(inferred_data_B).append(event_data_B).append(explanation_data_B)
all_data_B.to_csv("../data/graphs/all_data_B.csv")
all_data = all_data_A.append(all_data_B)
all_data.to_csv("../data/graphs/all_data_A.csv")

all_triples = TriplesFactory.from_labeled_triples(all_data[["s", "p", "o"]].values)

with open('../data/splits/entity_to_id.pkl', 'wb') as f:
    pickle.dump(all_triples.entity_to_id, f)
    
with open('../data/splits/relation_to_id.pkl', 'wb') as f:
    pickle.dump(all_triples.relation_to_id, f)


# In[23]:


from sklearn.model_selection import train_test_split

training_data = time_data.append(topology_data).append(inferred_data).append(event_data).append(explanation_data)
test_vali_data = inferred_data.append(explanation_data)
test_data, validation_data = train_test_split(test_vali_data, test_size=0.5, random_state=0)

training_data.to_pickle('../data/splits/training.pkl')
test_data.to_pickle('../data/splits/test.pkl')
validation_data.to_pickle('../data/splits/validation.pkl')

all_data = time_data.append(topology_data).append(inferred_data).append(event_data).append(explanation_data)
all_data.to_csv("../data/graphs/all_data_A.csv")

