:: ## GraphDB needs to be open and containing the right files and graphs

:: convert jupyter notebooks to .py files
jupyter nbconvert --to script 01.1_Preprocessing.ipynb
jupyter nbconvert --to script 02.1_TrainTransE.ipynb
jupyter nbconvert --to script 02.02_TrainComplEx.ipynb
jupyter nbconvert --to script 02.03_TrainTransH.ipynb
jupyter nbconvert --to script 02.04_TrainTTransE.ipynb
jupyter nbconvert --to script 03.1_Evaluation.ipynb
: 03.2_PerformanceMetrics is a jupyter notebook, which produces many charts to analyse model performance
jupyter nbconvert --to script src/code/04_LinkPrediction.ipynb

:: run preprocessing (needs to be run on local computer, where GraphDB is running)
python 01.1_Preprocessing.py

if NOT ["%errorlevel%"]==["0"] pause

:: transfer data, code  and requirements.txt to server
scp src/requirements.txt escher:~/src/
scp -r src/data/splits/ escher:~/src/data/
scp -r src/data/graphs/ escher:~/src/data/
scp src/code/*.py escher:~/src/code/
scp -r src/code/TKGE/ escher:~/src/code/


:: install requirements on escher and run models
:: detach: Strg+A, D

ssh escher 
screen -S evaluation
. venv/thesis/bin/activate
cd src
pip install requirements.txt
cd code
python 02.01_TrainTransE.py 02.02_TrainComplEx.py 02.03_TrainTransH.py

cd TKGE

python tkge.py train --config "./ExpCPS_config/TTransE_ExpCPS.yaml"
python tkge.py eval --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment/ttranse/ExpCPS/ex000000

python tkge.py hpo --config "./ExpCPS_config/TTransE_ExpCPSHPO.yaml"
python tkge.py hpo --resume "./ExpCPS_config/TTransE_ExpCPSHPO.yaml"

python tkge.py eval --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO/ttranse/ExpCPS/ex000001/trial17

python tkge.py eval --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001

cd src/code/TKGE
python -m debugpy --listen 5678 

python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestRejectedEvent:668ba3c2-30cd-46ba-89dd-f4b57c28d6f8 -ts 1973-12-25T04:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestRejectedEvent:477136ed-0327-4bd2-a8be-fc29278674a8 -ts 1973-12-25T05:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:OverloadingEvent:60d6ca9e-61d1-40bd-8dfa-aaca775e7c2e -ts 1973-12-25T05:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:NormalizingEvent:1fd24054-f754-4ff6-8bb1-752f84a1f893 -ts 1973-12-25T08:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:LoweringDemandEvent:a6d24c3d-8d27-4c28-8296-85c6d53af701 -ts 1973-12-25T08:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:17398359-f908-4703-84e5-e4f5679c31d7 -ts 1973-12-25T11:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:7533e192-eaa4-4d12-baed-c4d5125e49ea -ts 1973-12-25T12:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:463fd27b-7be1-44f8-8048-3b0303f9e222 -ts 1973-12-25T13:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:235cb939-67eb-498c-96dc-1a0ecc2b460a -ts 1973-12-25T14:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:PeakingDemandEvent:7be7cfb0-fce2-4333-bf78-a6de7171b24f -ts 1973-12-25T15:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:8ed9fd7c-84c1-4d86-ab92-fc36bc602e74 -ts 1973-12-25T15:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexRequestApprovedEvent:e54828ad-cb3e-4f7d-976e-3212ec345c43 -ts 1973-12-25T16:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:OverloadingEvent:b07057e0-b8a1-4d50-b721-02b38837b043 -ts 1973-12-25T18:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexUnavailableState:39f4a6c5-816d-4d15-ae18-0438348604bd -ts 1973-12-25T21:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexUnavailableState:f041fe99-ce1c-45b3-851e-5195a36ab664 -ts 1973-12-25T22:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexUnavailableState:e3e8bff1-c5f5-4a0a-bea1-b19ef2f62793 -ts 1973-12-25T23:00:40
python tkge.py pred --ex ./ExpCPS_checkpoints/TTransE_ExpCPS_experiment_HPO_bestModel/ttranse/ExpCPS/ex000001 -head expcps:FlexUnavailableState:86a82226-1cba-4f12-a5bb-6b6ccfc4c47f -ts 1973-12-26T00:00:40

python tkge.py train --config "./ExpCPS_config/config.yaml"


python 02.04_TrainTTransE.py

::try ttranse

if NOT ["%errorlevel%"]==["0"] pause

python 03.1_Evaluation.py
exit
exit

if NOT ["%errorlevel%"]==["0"] pause

:: run all training files and evaluation in one command:
:: ipython 02.01_TrainTransE.py 02.02_TrainComplEx.py 02.03_TrainTransH.py 02.04_TrainTTransE.py 03.1_Evaluation.py

:: get model results and trained models from escher
scp -r escher:~/src/models src/
scp -r escher:~/src/model_results src/

:: continue with 03.2_PerformanceMetrics.ipynb

:: continue with 04_LinkPrediction
:: ipython 04_LinkPrediction.py




ssh escher
nvidia-smi
kill -SIGHUP 609
nvidia-smi
exit

jupyter nbconvert --to notebook --execute src/code/Untitled1.ipynb
