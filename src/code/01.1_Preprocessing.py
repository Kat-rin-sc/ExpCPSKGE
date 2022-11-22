#!/usr/bin/env python
# coding: utf-8

# # Code for Preprocessing of Knowledge Graph
# 
# Only run this script locally, where the right data is imported to GraphDB and GraphDB is running

# ## Setup and load libraries

# In[13]:


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
import re
from tqdm import tqdm  


# ## load Data using SPARQL
# 
# ### Graph Names

# In[24]:


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
def get_query_result(query, query_name, repository):
    # repository source
    endpoint = "http://localhost:7200/repositories/"+repository

    df = sparql_dataframe.get(endpoint, query, post = True)

    # replace namespaces by prefixes
    df = df.replace("http://expcps.link/vocab#", "expcps:", regex=True)
    df = df.replace("http://www.w3.org/2003/01/geo/wgs84_pos#", "wgs:", regex=True)
    df = df.replace("http://expcps.link/expcps/SoftProtection#", "test:", regex=True)
    df = df.replace("http://expcps.link/expcps/SoftProtectionV2#", "test:", regex=True)
    df = df.replace("http://expcps.link/expcps/OverloadSupport#", "test:", regex = True)
    df = df.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:", regex=True)
    df = df.replace("http://www.w3.org/2002/07/owl#", "owl:", regex=True)
    df = df.replace("https://w3id.org/function/ontology#", "fno:", regex=True)
    df = df.replace("http://www.geonames.org/ontology#", "gn:", regex=True)
    df = df.replace("http://www.w3.org/2001/XMLSchema#", "xsd:", regex=True)
    df = df.replace("http://www.w3.org/2000/01/rdf-schema#", "rdfs:", regex=True)
    df = df.replace("http://www.w3.org/ns/sosa/", "sosa:", regex=True)
    df = df.replace("http://expcps.link/expcps/properties#", "prop:", regex=True)

    if 'p' in df.columns:
        df = df.loc[df.p != "sosa:hasSimpleResult"]
        df = df.loc[df.p != "expcps:refValues"]

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

graph_names_A = get_query_result(graph_names_query, "graphNames", "VillageA")

graph_names_B = get_query_result(graph_names_query, "graphNames", "VillageC")


# ### time_data 
# 
# -  all timeseries data points
# 
# Time Data is all the measurements in the Knowledge Graph, which are connected to a Timestamp. As the data for VillageA and VillageB are not exactly in the timeframe which we would like, all timestamps above the timeframe we want to measure are cut. Also any timestamps from before the start of the simulation are deleted as these triples are not connected to an actual timestamp(are already there when the simulation is starting).
# 
# #### Village A
# 
# The Original Data of Village A contained data from 1AM on 1973-12-25 to 4AM the next day. Any timestamps before this should be deleted and the data will be cut to include only one day.

# In[3]:


# get timeseries data
time_graphs = graph_names_A.loc[graph_names_A.graphName.str.startswith("http://expcps.link/update/SoftProtection")]

source = "FROM <" + time_graphs["graphName"].str.cat( sep = ">\nFROM <") + ">"
# build query
q = prefixes + "select *" + source + "where{?s ?p ?o}"

time_data_A = get_query_result(q, "time_data_A", "VillageA")
time_data_A.head()


# In[4]:


time_data_A.shape


# In[5]:


# get all events which have a timestamp after 1973-12-26T00:00:40

all_timestampsA = time_data_A.loc[time_data_A.p == "sosa:resultTime"].o.unique()
all_timestamps_splitA = np.array([re.split(r'[-T:]',x) for x in all_timestampsA.ravel()]).astype(int)

early_timestampsA = all_timestampsA[(all_timestamps_splitA[:,0]<1974)&((all_timestamps_splitA[:,2]<25)|(all_timestamps_splitA[:,1]<12))]
late_timestampsA = all_timestampsA[(all_timestamps_splitA[:,0]>=1973)&(all_timestamps_splitA[:,1]>=12)&(all_timestamps_splitA[:,2]>25)]
used_timestampsA = all_timestampsA[(all_timestamps_splitA[:,0]==1973)&(all_timestamps_splitA[:,1]==12)&(all_timestamps_splitA[:,2]==25)]
late_observationsA = np.array(time_data_A.loc[time_data_A.o.isin(late_timestampsA) & (time_data_A.p == "sosa:resultTime")].s)


# In[7]:


# delete all late observations and early timestamps
time_data_A = time_data_A.loc[~time_data_A.s.isin(late_observationsA)]
time_data_A = time_data_A.loc[~time_data_A.o.isin(early_timestampsA)]

time_data_A.to_csv('../data/graphs/time_data_A.csv')


# #### Village B

# In[25]:


# get timeseries data
time_graphs = graph_names_B.loc[graph_names_B.graphName.str.startswith("http://expcps.link/update/SoftProtectionV2")]

source = "FROM <" + time_graphs["graphName"].str.cat( sep = ">\nFROM <") + ">"
# build query
q = prefixes + "select *" + source + "where{?s ?p ?o}"

time_data_B = get_query_result(q, "time_data_B", "VillageC")
time_data_B.head()


# In[13]:


# get all events which have a timestamp after 1974-2-13/14

all_timestampsB = time_data_B.loc[time_data_B.p == "sosa:resultTime"].o.unique()
all_timestamps_splitB = np.array([re.split(r'[-T:]',x) for x in all_timestampsB.ravel()]).astype(int)

early_timestampsB = all_timestampsB[(all_timestamps_splitB[:,0]<1974)&((all_timestamps_splitB[:,2]<25)|(all_timestamps_splitB[:,1]<12))]
late_timestampsB = all_timestampsB[(all_timestamps_splitB[:,0]>=1973)&(all_timestamps_splitB[:,1]>=12)&(all_timestamps_splitB[:,2]>25)]
used_timestampsB = all_timestampsB[(all_timestamps_splitB[:,0]==1973)&(all_timestamps_splitB[:,1]==12)&(all_timestamps_splitB[:,2]==25)]
#used_timestampsB2 = all_timestampsB[(all_timestamps_splitB[:,0]==1974) & (all_timestamps_splitB[:,1]==2) & (all_timestamps_splitB[:,2]==14)]
late_observationsB = np.array(time_data_B.loc[time_data_B.o.isin(late_timestampsB) & (time_data_B.p == "sosa:resultTime")].s)
#used_observationsB1 = np.array(time_data_B.loc[time_data_B.o.isin(used_timestampsB1) & (time_data_B.p == "sosa:resultTime")].s)
#used_observationsB2 = np.array(time_data_B.loc[time_data_B.o.isin(used_timestampsB2) & (time_data_B.p == "sosa:resultTime")].s)


# In[14]:


time_data_B = time_data_B.loc[~time_data_B.s.isin(late_observationsB)]
time_data_B = time_data_B.loc[~time_data_B.o.isin(early_timestampsB)]
#time_data_B1 = time_data_B.loc[~time_data_B.s.isin(used_observationsB2)]
#time_data_B2 = time_data_B.loc[~time_data_B.s.isin(used_observationsB1)]

time_data_B.to_csv('../data/graphs/time_data_B.csv')
#time_data_B1.to_csv('../data/graphs/time_data_B1.csv')
#time_data_B2.to_csv('../data/graphs/time_data_B2.csv')


# ### topology_data
# 
# - setup of Knowledge Graph (Ontology)
# #### Village A

# In[16]:


# get topology data
topology_input = graph_names_A.loc[graph_names_A.graphName.isin(["http://expcps.link/topology/SoftProtection"])]
source = "FROM <" + topology_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
topology_data_A = get_query_result(q, "topology_data_A", "VillageA")
topology_data_A.head()


# In[17]:


topology_data_A.shape


# #### Village B

# In[26]:


# get topology data
topology_input = graph_names_B.loc[graph_names_B.graphName.str.startswith("http://expcps.link/topology/SoftProtectionV2")]
source = "FROM <" + topology_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
topology_data_B = get_query_result(q, "topology_data_B", "VillageC")
topology_data_B.head()


# In[27]:


topology_data_B.shape


# ### inferred_data
# 
# - potential causalities between observations
# #### Village A

# In[29]:


# get inferred data from causality relations
inferred_input = graph_names_A.loc[graph_names_A.graphName.str.startswith("http://expcps.link/inferred")]
source = "FROM <" + inferred_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
inferred_data_A = get_query_result(q, "inferred_data_A", "VillageA")
inferred_data_A.head()


# In[30]:


inferred_data_A.shape


# #### Village B

# In[31]:


# get inferred data from causality relations
inferred_input = graph_names_B.loc[graph_names_B.graphName.str.startswith("http://expcps.link/inferred/SoftProtectionV2")]
source = "FROM <" + inferred_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
inferred_data_B = get_query_result(q, "inferred_data_B", "VillageC")
inferred_data_B.head()


# In[32]:


inferred_data_B.shape


# ### events_data 
# 
# - all events based on some criteria
# #### Village A

# In[33]:


# get events data
event_input = graph_names_A.loc[graph_names_A.graphName.str.startswith("http://expcps.link/event")]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
event_data_A = get_query_result(q, "event_data_A", "VillageA")
event_data_A.head()


# In[34]:


event_data_A.shape


# In[35]:


late_events_A = np.array(event_data_A.loc[event_data_A.o.isin(late_observationsA)].s.unique())
event_data_A =event_data_A.loc[~event_data_A.s.isin(late_events_A)]

event_data_A.to_csv('../data/graphs/event_data_A.csv')


# In[36]:


event_data_A.shape


# In[37]:


event_data_A.loc[event_data_A.p == "rdf:type"].groupby("o").count().sort_index()


# In[38]:


len(event_data_A.s.unique())


# #### Village B

# In[39]:


# get events data
event_input = graph_names_B.loc[graph_names_B.graphName.str.startswith("http://expcps.link/event/SoftProtectionV2")]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
q = prefixes + "SELECT *\n" + source + "WHERE {?s ?p ?o}"
event_data_B = get_query_result(q, "event_data_B", "VillageC")
event_data_B.head()


# In[40]:


event_data_B.shape


# In[41]:


late_events_B = np.array(event_data_B.loc[event_data_B.o.isin(late_observationsB)].s.unique())
event_data_B = event_data_B.loc[~event_data_B.s.isin(late_events_B)]

event_data_B.to_csv('../data/graphs/event_data_B.csv')


# In[42]:


print(event_data_B.shape)


# ### explanation_data 
# 
# - two possible paths of events
# #### Village A

# In[45]:


# get events data
explanation_input = graph_names_A.loc[graph_names_A.graphName.str.startswith("http://expcps.link/explanation")]
source = "FROM <" + explanation_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
query1 = prefixes + "SELECT ?effect ?causedBy ?cause\n" + source + """
WHERE {?Argument expcps:causeEvent ?cause .
        ?Argument expcps:effectEvent ?effect}
"""
explanation_data_A = get_query_result(query1, "explanation_data_A", "VillageA")
explanation_data_A["causedBy"] = "expcps:causedBy"
explanation_data_A.columns = ["s", "p", "o"]
explanation_data_A.head()


# In[46]:


explanation_data_A.shape


# In[47]:


explanation_data_A = explanation_data_A.loc[~explanation_data_A.s.isin(late_events_A)]
explanation_data_A = explanation_data_A.loc[~explanation_data_A.o.isin(late_events_A)]
explanation_data_A.shape


# In[48]:


explanation_data_A.to_csv('../data/graphs/explanation_data_A.csv')


# In[49]:


explanation_data_A.groupby("s").count().sort_index()


# In[50]:


explanation_data_A.loc[explanation_data_A.s == "expcps:PeakingDemandEvent:fcc22e73-75da-44c7-9c31-9ca9e246700a"]


# #### Village B

# In[51]:


# get events data
event_input = graph_names_B.loc[graph_names_B.graphName.str.startswith("http://expcps.link/explanation/SoftProtectionV2")]
source = "FROM <" + event_input["graphName"].str.cat( sep = ">\nFROM <") + ">\n"
# build query
query1 = prefixes + "SELECT ?effect ?causedBy ?cause\n" + source + """
WHERE {?Argument expcps:causeEvent ?cause .
        ?Argument expcps:effectEvent ?effect}
"""
explanation_data_B = get_query_result(query1, "explanation_data_B", "VillageC")
explanation_data_B["causedBy"] = "expcps:causedBy"
explanation_data_B.columns = ["s", "p", "o"]
explanation_data_B.head()


# In[52]:


explanation_data_B.shape


# In[53]:


explanation_data_B = explanation_data_B.loc[~explanation_data_B.s.isin(late_events_B)]
# explanation_data_B1 = explanation_data_B.loc[~explanation_data_B.s.isin(used_events_B2)]
# explanation_data_B2 = explanation_data_B.loc[~explanation_data_B.s.isin(used_events_B1)]


# In[54]:


print(explanation_data_B.shape)
# print(explanation_data_B1.shape)
# print(explanation_data_B2.shape)


# In[55]:


explanation_data_B.to_csv('../data/graphs/explanation_data_B.csv')
# explanation_data_B1.to_csv('../data/graphs/explanation_data_B1.csv')
# explanation_data_B2.to_csv('../data/graphs/explanation_data_B2.csv')


# In[56]:


explanation_data_B.groupby("s").count().sort_index()


# ## Training, Test, Validation Splits
# 
# **Data available:**
# * **topology_data** - data on the topology of the graph, no timeseries data
# * **time_data** - all the time series data derived from BIFROST
# * **inferrred_data** - possible explanations inferred by the graph
# * **event_data** - detected events in the KG
# * **explanation_data** - all explanations for events based in the explanations graph, but reducing to caused_by relations

# In[57]:


from pykeen.triples import TriplesFactory
import pickle

all_data_A = time_data_A.append(topology_data_A).append(inferred_data_A).append(event_data_A).append(explanation_data_A)
all_data_A.to_csv("../data/graphs/all_data_A.csv")

all_data_B = time_data_B.append(
             topology_data_B).append(
             inferred_data_B).append(
             event_data_B).append(
             explanation_data_B)
all_data_B.to_csv("../data/graphs/all_data_B.csv")

all_data = all_data_A.append(all_data_B)
all_data.to_csv("../data/graphs/all_data_AB.csv")

all_triples = TriplesFactory.from_labeled_triples(all_data[["s", "p", "o"]].values)

with open('../data/splits/entity_to_id.pkl', 'wb') as f:
    pickle.dump(all_triples.entity_to_id, f)
    
with open('../data/splits/relation_to_id.pkl', 'wb') as f:
    pickle.dump(all_triples.relation_to_id, f)


# ## Create quadruples, add time information to triples

# In[34]:


def add_timestamps(triples, fileName, start_time):
    quads = triples.copy()
    quads["t"] = 0
    
    # set start time to all triples defining Type of entity 
    quads.loc[quads.p =="rdf:type", "t"] = start_time
    
    # for all subjects which have a resultTime, set timestamps to this time
    subjects = quads.s.unique()
    for s in tqdm (subjects, desc="Adding direct Timestamps to subjects ..."):
        if not quads.loc[(quads.s == s) & (quads.p == "sosa:resultTime")].empty:
            timestamp = quads.loc[(quads.s == s) & (quads.p == "sosa:resultTime"),"o"].values[0]
            quads.loc[(quads.s == s), "t"] = timestamp
    
    # delete resulttime triples
    # quads = quads.loc[quads.p != "sosa:resultTime"]
    # set start time to any remaining triples
    quads.loc[quads.t ==0, "t"] = start_time
    quads.to_csv("../data/graphs/" + fileName + ".csv")
    return quads

def add_timestamps_start(triples, fileName, start_time):
    quads = triples.copy()
    quads["t"] = start_time
    
    # delete resulttime triples
    quads = quads.loc[quads.p != "sosa:resultTime"]
    quads.to_csv("../data/graphs/" + fileName + ".csv")
    return quads

def add_timestamps_event(triples, timestampTriples, fileName, start_time):
    quads = triples.copy()
    quads["t"] = 0
    
    timestampTriples = timestampTriples.loc[timestampTriples.p == "sosa:resultTime"]
    
    # for all events, find referenced observation
    events = triples.s.unique()
    
    for event in tqdm (events, desc="Adding indirect Timestamps to events ..."):
        observation = triples.loc[(triples.s == event) & (triples.p == "expcps:hasReference")].o.values[0]
        timestamp = timestampTriples.loc[timestampTriples.s == observation].o.values[0]
        quads.loc[(quads.s == event), "t"] = timestamp
        
    # delete resulttime triples
    # quads = quads.loc[quads.p != "sosa:resultTime"]
        
    # set start time to any remaining triples
    quads.loc[quads.t ==0, "t"] = start_time
    quads.to_csv("../data/graphs/" + fileName + ".csv")
    return quads

def add_timestamps_explanation(exp_data, event_data, fileName):
    # for all events, find referenced observation
    events = exp_data.s.unique()
    quads = exp_data.copy()
    quads["t"] = 0
    
    for event in tqdm (events, desc="Adding indirect Timestamps to explanations ..."):
        ts = event_data.loc[(event_data.s == event)].t.values[0]
        quads.loc[(quads.s == event), "t"] = ts     
        
    quads.to_csv("../data/graphs/" + fileName + ".csv")
    return quads


# In[59]:


start_A = all_data_A.loc[all_data_A.p == "sosa:resultTime"].o.min()
start_B = all_data_B.loc[all_data_B.p == "sosa:resultTime"].o.min()

time_data_A_4 = add_timestamps(time_data_A, "time_data_A_quads", start_A)
time_data_B_4 = add_timestamps(time_data_B, "time_data_B_quads", start_B)

topology_data_A_4 = add_timestamps_start(topology_data_A, "topology_data_A_quads", start_A)
topology_data_B_4 = add_timestamps_start(topology_data_B, "topology_data_B_quads", start_B)

inferred_data_A_4 = add_timestamps_start(inferred_data_A, "inferred_data_A_quads", start_A)
inferred_data_B_4 = add_timestamps_start(inferred_data_B, "inferred_data_B_quads", start_B)


# In[21]:


from helper import *


# In[32]:


event_data_A_quads = pd.read_csv("../data/graphs/event_data_A_quads.csv", index_col = 0)
event_data_B_quads = pd.read_csv("../data/graphs/event_data_B_quads.csv", index_col = 0)
explanation_data_A = pd.read_csv("../data/graphs/explanation_data_A.csv", index_col = 0)
explanation_data_B = pd.read_csv("../data/graphs/explanation_data_B.csv", index_col = 0)


# In[60]:


event_data_A_4 = add_timestamps_event(event_data_A, time_data_A, "event_data_A_quads", start_A)
event_data_B_4 = add_timestamps_event(event_data_B, time_data_B, "event_data_B_quads", start_B)

explanation_data_A_4 = add_timestamps_explanation(explanation_data_A, event_data_A_quads, "explanation_data_A_quads")
explanation_data_B_4 = add_timestamps_explanation(explanation_data_B, event_data_B_quads, "explanation_data_B_quads")


# # Create train and test splits
# 
# * **Split1_NT** - triples, no Time data
#     * *training1NT*: all of Village A, topology, timestamps, events of Village B
#     * *valid1NT*: inferred, explanation of Village B
# 
# * **Split1_T** - triples, no Time data
#     * *training1T*: all of Village A, topology, timestamps, events of Village B
#     * *valid1T*: inferred, explanation of Village B
# 
#     
# training is then split randomly in training and test data for training purposes.
#     

# In[61]:


from sklearn.model_selection import train_test_split

# split 1: 
trainingNT = time_data_A.append(
    topology_data_A).append(
    inferred_data_A).append(
    event_data_A).append(
    explanation_data_A).append(
    topology_data_B).append(
    time_data_B).append(
    event_data_B).append(
    inferred_data_B)

trainingNT, testNT = train_test_split(trainingNT, test_size=0.2, random_state=0)
testNT, test_validNT = train_test_split(testNT, test_size=0.5, random_state=0)

validNT = explanation_data_B

trainingNT.to_pickle('../data/splits/trainingNT.pkl')
testNT.to_pickle('../data/splits/testNT.pkl')
test_validNT.to_pickle('../data/splits/test_validNT.pkl')
validNT.to_pickle('../data/splits/validNT.pkl')


# In[38]:


time_data_A_4 = pd.read_csv("../data/graphs/time_data_A_quads.csv", index_col = 0)
topology_data_A_4 = pd.read_csv("../data/graphs/topology_data_A_quads.csv", index_col = 0)
inferred_data_A_4 = pd.read_csv("../data/graphs/inferred_data_A_quads.csv", index_col = 0)
event_data_A_4 = pd.read_csv("../data/graphs/event_data_A_quads.csv", index_col = 0)
explanation_data_A_4 = pd.read_csv("../data/graphs/explanation_data_A_quads.csv", index_col = 0)

time_data_B_4 = pd.read_csv("../data/graphs/time_data_B_quads.csv", index_col = 0)
topology_data_B_4 = pd.read_csv("../data/graphs/topology_data_B_quads.csv", index_col = 0)
inferred_data_B_4 = pd.read_csv("../data/graphs/inferred_data_B_quads.csv", index_col = 0)
event_data_B_4 = pd.read_csv("../data/graphs/event_data_B_quads.csv", index_col = 0)
explanation_data_B_4 = pd.read_csv("../data/graphs/explanation_data_B_quads.csv", index_col = 0)

all_data_B_4 = time_data_B_4.append(
             topology_data_B_4).append(
             inferred_data_B_4).append(
             event_data_B_4).append(
             explanation_data_B_4)
all_data_B_4.to_csv("../data/graphs/all_data_B_quads.csv")


# In[39]:


# split 1 with timestamps: 
trainingT = time_data_A_4.append(
    topology_data_A_4).append(
    inferred_data_A_4).append(
    event_data_A_4).append(
    explanation_data_A_4).append(
    topology_data_B_4).append(
    time_data_B_4).append(
    event_data_B_4).append(
    inferred_data_B_4)

trainingT, testT = train_test_split(trainingT, test_size=0.2, random_state=0)
testT, test_validT = train_test_split(testT, test_size=0.5, random_state=0)

validT = explanation_data_B_4

trainingT.to_pickle('../data/splits/trainingT.pkl')
testT.to_pickle('../data/splits/testT.pkl')
test_validT.to_pickle('../data/splits/test_validT.pkl')
validT.to_pickle('../data/splits/validT.pkl')


# In[40]:


trainingT.head()


# In[41]:


trainingNT.head()


# In[66]:


time_data_A.s


# In[67]:


time_data_A_4.s


# ## Move temporal quadruples to model folder

# In[42]:


import pickle as pkl
import pandas as pd
import os
import shutil


# In[43]:


training = pd.read_pickle(r"../data/splits/trainingT.pkl")
training = training.loc[training.p != "expcps:refValues"]
training.to_csv("TKGE/data/ExpCPS/train.txt", index = False, sep = "\t", header = False)

test = pd.read_pickle(r"../data/splits/testT.pkl")
test = test.loc[test.p != "expcps:refValues"]
test.to_csv("TKGE/data/ExpCPS/test.txt", index = False, sep = "\t", header = False)

valid = pd.read_pickle(r"../data/splits/validT.pkl")
valid = valid.loc[valid.p != "expcps:refValues"]
valid.to_csv("TKGE/data/ExpCPS/valid.txt", index = False, sep = "\t", header = False)


# In[44]:


inferred = pd.read_csv("../data/graphs/inferred_data_B_quads.csv", index_col = 0)
inferred.to_csv("TKGE/data/ExpCPS/inferred.txt", index = False, sep = "\t", header = False)

explanation = pd.read_csv("../data/graphs/explanation_data_B_quads.csv", index_col = 0)
explanation.to_csv("TKGE/data/ExpCPS/explanation.txt", index = False, sep = "\t", header = False)

events = pd.read_csv("../data/graphs/event_data_B_quads.csv", index_col = 0)
events.to_csv("TKGE/data/ExpCPS/events.txt", index = False, sep = "\t", header = False)

all_data = pd.read_csv("../data/graphs/all_data_B_quads.csv", index_col = 0)
all_data.to_csv("TKGE/data/ExpCPS/all.txt", index = False, sep = "\t", header = False)

