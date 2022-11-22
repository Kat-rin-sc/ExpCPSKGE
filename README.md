# ExpCPSKGE
Code to reproduce the results of the thesis "Causality Prediction in a Cyber-Physical-Energy-System using Knowledge Graph Embeddings".

## Data 

For running the whole embedding pipeline, a GraphDB instance needs to be running, containing the Knowledge Graphs which should be analysed.
The names of the repositories of the two village to be analysed need to be "VillageA" and "VillageC".

## Workflow

When conducting the next steps in order, the experiment should run through and the results presented in the thesis will be obtained.

0. Install Requirements, as stated in requirements.txt

1. Create data dump, containing preprocessed data
	- run 01.1_Preprocessing.py 

(optional: Transfer data and models to a GPU)

2. Train and evaluate models (should be run on a GPU)
	- run 02.01_TrainTransE.py
	- run 02.02_TrainComplEx.py
	- run 02.03_TrainTransH.py
	- run 02.04_TrainTTransE.py
	- run 3.1_Evaluation.py 

3. Use models for Link Prediction
	- run 04.1_LinkPrediction.py

3. Analyse and Interpret model Performance, Link Prediction Results, etc. The results can be found in the following Jupyter Notebooks:
	- 04.2_PredictionEvaluation.ipynb
	- 04.3_AnalyseCausalities.ipynb
	- ...
