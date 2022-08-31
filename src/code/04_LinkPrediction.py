#!/usr/bin/env python
# coding: utf-8

# # 4. Link Prediction to find new Relation Paths between Events

# In[1]:


from helper import *


# ## Import ground truth data

# In[2]:


explanation_graph_B = pd.read_csv("../data/graphs/explanation_data_B1.csv", index_col = 0)
explanation_graph_B.shape
explanation_graph_B.columns = ["s","p","o"]


# In[3]:


print("Number of Events in the Data:")
events = explanation_graph_B.s.unique()
print(len(explanation_graph_B.s.unique()))
print("First Event:", explanation_graph_B.s[16] )


# In[4]:


# plt.tight_layout()
# nx.draw_networkx(g1, arrows=True, with_labels = False)
# plt.show()
# # tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
# plt.clf();


# In[5]:


n_causes = []
for event_name in events:
    explanation = get_explanation_path(event_name, explanation_graph_B)
    causes = list(explanation.predecessors(event_name))
    n = len(causes)
    n_causes.append(n)
print(n_causes)


# ## Analyse an event in detail

# In[6]:


top_n = 5
model_name = "TransE_Adam_hpo_100EbestModel"
event_name = "expcps:FlexRequestApprovedEvent:d9bf1a4e-dd6b-4514-a816-5db056b2e8eb"


# In[7]:


# get ground truth
explanation = get_explanation_path(event_name, explanation_graph_B)
causes = list(explanation.predecessors(event_name))
effects = list(explanation.successors(event_name))


# In[22]:


# get data about Event
all_data_B = pd.read_csv("../data/graphs/all_data_AB.csv", index_col = 0)

def get_info_about(event_name, data):
    print(event_name)
    eventType = data.loc[(data.s == event_name) & (data.p == "rdf:type") & (data.o != "expcps:Event")].o.item()
    madeBySensor = data.loc[(data.s == event_name) & (data.p == "sosa:madeBySensor")].o.item()
    hasReference = data.loc[(data.s == event_name) & (data.p == "expcps:hasReference")].o.item()
    hasLocation =  data.loc[(data.s == event_name) & (data.p == "expcps:hasLocation")].o.item()
    resultTime =   data.loc[(data.s == hasReference)&(data.p == "sosa:resultTime")].o.item()
    
        
    return {#"label": event_name,
            "eventType": eventType, 
            "madeBySensor" : madeBySensor, 
            "hasReference" : hasReference, 
            "hasLocation" : hasLocation, 
            "resultTime" : resultTime
           }


# In[23]:


available_events = list(all_data_B.loc[(all_data_B.p == "rdf:type") & (all_data_B.o.str.contains("Event"))].s.unique())


# In[24]:


# get model predictions
# calculate predictions
model = load_model(model_name)
training = get_triples('../data/splits/trainingNT.pkl')
x = 1
for event_name in events:
    print(event_name)
    predicted_tails_df = predict.get_tail_prediction_df(
        model, 
        head_label = event_name, 
        relation_label = 'expcps:causedBy', 
        triples_factory=training,
    )
    # only keep top n predictions and true tails
    predicted_tails_df["in_data"] = (predicted_tails_df.tail_label.isin(causes))
    predicted_tails_df["rank"] = predicted_tails_df["score"].rank(ascending =False)
    predicted_tails_df.drop('tail_id', axis=1, inplace=True)
    predicted_tails_df.drop('in_training', axis=1, inplace=True)

    top_preds = predicted_tails_df.nlargest(top_n, columns=['score'])
    true_tails = predicted_tails_df.loc[predicted_tails_df.in_data == True]
    top_preds = pd.concat([top_preds, true_tails]).reset_index(drop = True)

    # add original event
    orig_event = pd.DataFrame({'tail_label': event_name, 'score': 0, 'in_data':"effectEvent",
                            'rank':0},index =[0])
    top_preds = pd.concat([orig_event, top_preds]).reset_index(drop = True)
    top_preds.rename(columns = {'tail_label':'label'}, inplace = True)

    # add event info for events
    event_info = pd.DataFrame()
    for i in top_preds.label:
        event_info = event_info.append(get_info_about(i, all_data_B),ignore_index=True)

    top_preds = top_preds.join(event_info)
    top_preds = top_preds.replace('http://expcps.link/expcps/OverloadSupport#','', regex=True)

    top_preds.to_csv("../predictions/top_"+str(top_n)+"_event"+str(x)+".csv")
    x +=1


# In[18]:


event_name = "expcps:FlexContributedEvent:52125454-fa4d-40c6-b38f-53024ed5c32b"
data = all_data_B
data = pd.read_csv("../data/graphs/all_data_AB.csv", index_col = 0)


# In[19]:


data.loc[(data.s == event_name)]


# In[21]:


eventType = data.loc[(data.s == event_name) & (data.p == "rdf:type") & (data.o != "expcps:Event")].o.item()
eventType


# In[ ]:


help(predict)


# In[ ]:





# In[ ]:


# check performance of model for path prediction
check_performance(top_preds, causes)


# In[ ]:


predicted_tails_df.loc[predicted_tails_df.in_training == True]


# In[ ]:


causes


# In[ ]:


predicted_tails_df["in_data"] = (predicted_tails_df.tail_label.isin(causes))
predicted_tails_df


# In[ ]:


predicted_tails_df.loc[predicted_tails_df.in_data == True]


# In[ ]:


explanation_graph_B.loc[explanation_graph_B.s == event_name]


# In[ ]:




