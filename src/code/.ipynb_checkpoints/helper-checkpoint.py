import os
import pickle
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import predict

import sparql_dataframe
from sklearn.model_selection import train_test_split
import networkx as nx




#---------------------------------------------------------------------------------------------
# from 02_BuildModelsKGE
entity_to_id = None
relation_to_id = None
    
def get_triples(fileName):
    global entity_to_id, relation_to_id
    if entity_to_id == None:
        with open('../data/splits/entity_to_id.pkl', 'rb') as f:
            entity_to_id = pickle.load(f)
            
    if relation_to_id == None:
        with open('../data/splits/relation_to_id.pkl', 'rb') as f:
            relation_to_id = pickle.load(f) 
    
    if fileName.split(".")[-1] == "pkl":
        df = pd.read_pickle(fileName)
    else:
        df = pd.read_csv(fileName)
    triples = TriplesFactory.from_labeled_triples(df[["s", "p", "o"]].values,
                                              entity_to_id=entity_to_id,
                                              relation_to_id=relation_to_id,)
    return triples

#--------------------------------------------------------------------------------------------
# from 03.1_Evaluation

def load_model(model_name):
    if not os.path.exists("../models/" + model_name): 
        print("Model does not exist/has not been trained: " + str(model_name))
        return
    if torch.cuda.is_available():
        model = torch.load("../models/" + model_name + "/trained_model.pkl")
    else:
        model = torch.load("../models/" + model_name + "/trained_model.pkl", map_location=torch.device('cpu'))
    return model

def evaluate_model_on(model, test_triples,all_triples_A, saveAs):
    evaluator = RankBasedEvaluator(filtered=True)

    results_TransE_default = evaluator.evaluate(
        model = model,
        mapped_triples=test_triples.mapped_triples,
        additional_filter_triples=[
            all_triples_A.mapped_triples,
        ],
    )

    results_df = results_TransE_default.to_df()
    results_df.to_csv("../model_results/" + saveAs + ".csv")
    return

#---------------------------------------------------------------------------------------------
# from 03.2_performanceMetrics

def reset_results():
    model = []
    mode = []
    evaluation = []
    data = []
    return model, mode, evaluation, data

def add_results(fileName, model, mode, evaluation, data):
    data_tmp = pd.read_csv("../model_results/" + fileName, index_col = 0)
    names = fileName.split("_")
    model.append(names[0])
    mode.append(names[1].lower())
    if len(names) > 2:
        evaluation.append(names[2].split(".")[0].lower())
    else:
        evaluation.append("training")
    data.append(data_tmp)
    return

def get_metric(metric, Side, Type, data):
    result = []
    for res in data:
        result.append(res.loc[(data[0].Metric == metric) &
                              (res.Side == Side) & 
                              (res.Type == Type)]["Value"].values[0]) 
    return result

#---------------------------------------------------------------------------------------------
# from 04_LinkPrediction

def get_explanation_path(event, graph):
    explanation = nx.DiGraph()
    find_cause(event, graph, explanation)
    find_effect(event, graph, explanation)
   # print("-------------")
    return explanation

def add_edge(explanation, cause, event):
    explanation.add_edges_from([(cause, event)])
   # print(event, "caused by", cause)
    return
    
def find_cause(event, graph, explanation):
    triple = graph.loc[graph.s == event]
    for index, row in triple.iterrows():
        add_edge(explanation, row.o, row.s)
        find_cause(row.o, graph, explanation)
        
def find_effect(event, graph, explanation):
    triple = graph.loc[graph.o == event]
    for index, row in triple.iterrows():
        add_edge(explanation, row.o, row.s)
        find_effect(row.s, graph, explanation)
        
def check_performance(predictions, truth):
    print("There are", len(truth), "causes in the ground truth data.")
    print("-------------")
    for i in range(len(predictions)):
        if predictions.values[i] in truth:
            print("Top", i+1, "exists in ground truth")
        else: 
            print("Top", i+1, "new potential cause:", predictions.values[i])