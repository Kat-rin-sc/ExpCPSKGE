import torch

import time
import os
from collections import defaultdict
import argparse

import pandas as pd
import numpy as np

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.pred.scoring import Prediction


class PredictTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Predict on a model"""
        subparser = parser.add_parser("pred", description=description, help="predict on a model.")

        subparser.add_argument(
            "-e",
            "--ex",
            type=str,
            help="specify the experiment folder",
            dest='experiment'
        )

        subparser.add_argument(
            "--checkpoint",
            type=str,
            default="best.ckpt",
            dest="ckpt_name",
            help="choose the checkpoint name in experiment folder from which training will be resumed."
        )

        subparser.add_argument(
            "-head",
            "--head",
            type=str,
            help="specify the head of the triple to be predicted",
            dest='head_target'
        )

        subparser.add_argument(
            "-rel",
            "--rel",
            type=str,
            help="specify the relation to be predicted",
            dest='rel_target'
        )
        subparser.add_argument(
            "-ts",
            "--timestamp",
            type=str,
            help="specify the timestamp to be predicted",
            dest='ts_target'
        )
        subparser.add_argument(
            "--topK",
            type=int,
            default="10",
            dest="topK",
            help="Choose the number of top predictions which should be returned"
        )

        return subparser

    def __init__(self, config: Config, ckpt_path: str, head_target: str, rel_target: str, ts_target: str, topK: int):
        super(PredictTask, self).__init__(config=config)
        self.ckpt_path = ckpt_path
        self.head_target = head_target
        self.rel_target = rel_target
        self.ts_target = ts_target
        self.topK = topK

        self.dataset = self.config.get("dataset.name")
        self.test_loader = None
        self.sampler = None
        self.model = None
        self.prediction = None

        self.test_bs = self.config.get("train.valid.batch_size")
        self.test_sub_bs = self.config.get("train.valid.subbatch_size") if self.config.get(
            "train.valid.subbatch_size") else self.test_bs

        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self.ckpt = torch.load(ckpt_path)

        self._prepare()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}")
        self.dataset = DatasetProcessor.create(config=self.config, head_target = self.head_target, rel_target = self.rel_target, ts_target = self.ts_target)

        self.config.log(f"Loading testing split data for loading")
        # TODO(gengyuan) load params
        self.test_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get(split = "pred"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.test_bs
        )

        self.config.log(f"Loading model {self.config.get('model.type')} from {self.ckpt_path}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.load_state_dict(self.ckpt['state_dict'], strict=True)
        self.model.to(self.device)


        self.config.log(f"Initializing prediction")
        self.prediction = Prediction(config=self.config, dataset=self.dataset)

        # validity checks and warnings
        self.subbatch_adaptive = self.config.get("train.subbatch_adaptive")

        if self.test_sub_bs >= self.test_bs or self.test_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.valid.sub_batch_size={self.test_sub_bs} is greater or equal to "
                            f"train.valid.batch_size={self.test_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.test_sub_bs = self.test_bs

    def main(self):
        self.config.log("BEGIN Prediction")

        predictions = self.pred()

        predictions = predictions.sort_values(by='score', ascending=False)
        print(predictions[0:self.topK])

        print("Full results saved to ../../predictions/TTransE_"+self.config.get("dataset.name")+ "_"+self.head_target+"_"+self.rel_target+"_"+self.ts_target+".csv")

        predictions.to_csv("../../predictions/TTransE_"+self.config.get("dataset.name")+ "_"+self.head_target+"_"+self.rel_target+"_"+self.ts_target+".csv")



    def pred(self):
        with torch.no_grad():
            self.model.eval()

            counter = 0

            predictions = pd.DataFrame(columns = ["tail_label", "score"])
        
            for batch in self.test_loader:
                done = False

                while not done:
                    try:
                        bs = batch.size(0)
                        dim = batch.size(1)

                        # batch_predictions = []
                        # batch_predictions['tail_label'] = []
                        # batch_predictions['score'] = []

                        batch = batch.to(self.device)

                        counter += bs

                        for start in range(0, bs, self.test_sub_bs):
                            stop = min(start + self.test_sub_bs, bs)
                            query_subbatch = batch[start:stop]
                            subbatch_df = self._subbatch_forward_predict(query_subbatch)

                            #subbatch_df = pd.DataFrame({"tail_label" : list(subbatch_candidates), "score": subbatch_scores})
                            predictions = predictions.append(subbatch_df)
                            
                        done = True

                    except RuntimeError as e:
                        if ("CUDA out of memory" not in str(e) or not self.subbatch_adaptive):
                            raise e

                        self.test_sub_bs //= 2
                        if self.test_sub_bs > 0:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation reduced to {self.test_sub_bs}.",
                                level="warning")
                        else:
                            self.config.log(
                                f"CUDA out of memory. Subbatch size for validation cannot be further reduces.",
                                level="error")
                            raise e
                
                # predictions.update(batch_predictions)
                #print(predictions)


            del batch
            torch.cuda.empty_cache()

            return predictions

    def _subbatch_forward_predict(self, query_subbatch):
        bs = query_subbatch.size(0)
        # queries_head = query_subbatch.clone()[:, :-1]
        queries_tail = query_subbatch.clone()[:, :-1]

        # queries_head[:, 0] = float('nan')
        queries_tail[:, 2] = float('nan')
        

        candidates, scores = self.model.predict_candidates(queries_tail)
        candidates = pd.Series(candidates)
        candidates = self.dataset.index2entity(candidates)     
        scores = pd.Series(scores[0])
        
        df_data = {"tail_label": candidates,
        "score": scores}

        df = pd.concat(df_data,
               axis = 1)

        #subbatch_predictions = pd.DataFrame([candidates, scores], columns = ["tail_label", "score"])   

        torch.cuda.empty_cache()

        return df