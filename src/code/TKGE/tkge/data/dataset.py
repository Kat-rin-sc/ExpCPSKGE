import torch
from torch.utils.data.dataset import Dataset as PTDataset
import numpy as np

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.utils import get_all_days_of_year, get_all_days_between, create_year2id, get_pretreated_data, get_all_hours_of_days

import enum
import arrow
import pdb

import re
import pandas as pd

from abc import ABC, abstractmethod

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


class DatasetProcessor(ABC, Registrable, Configurable):
    def __init__(self, config: Config, head_target: str = "", rel_target: str="",ts_target: str=""):
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

        self.folder = self.config.get("dataset.folder")
        self.resolution = self.config.get("dataset.temporal.resolution")
        self.index = self.config.get("dataset.temporal.index")
        self.float = self.config.get("dataset.temporal.float")
        self.name = self.config.get("dataset.name")

        self.reciprocal_training = self.config.get("task.reciprocal_training")
        # self.filter_method = self.config.get("data.filter")

        self.train_raw = []
        self.valid_raw = []
        self.test_raw = []

        self.ent2id = defaultdict(None)
        self.id2ent = defaultdict(None)
        self.rel2id = defaultdict(None)
        self.id2rel = defaultdict(None)
        self.ts2id = defaultdict(None)

        self.train_set = defaultdict(list)
        self.valid_set = defaultdict(list)
        self.test_set = defaultdict(list)
        self.pred_set = defaultdict(list)

        self.inferred_set = defaultdict(list)
        self.explanation_set = defaultdict(list)
        self.events_set = defaultdict(list)
        self.all_set = defaultdict(list)

        self.all_triples = []
        self.all_quadruples = []

        self.head_target = head_target
        self.rel_target = rel_target
        self.ts_target = ts_target

        self.load()
        self.process()
        # self.filter()

    @classmethod
    def create(cls, config: Config, head_target: str="", rel_target: str="", ts_target: str=""):
        """Factory method for data creation"""
        
        ds_type: str = config.get("dataset.name")

        if ds_type in DatasetProcessor.list_available():
            kwargs = config.get("dataset.args")  # TODO: ????????????key?????????
            return DatasetProcessor.by_name(ds_type)(config, head_target = head_target, rel_target = rel_target, ts_target = ts_target)  # return an instance
        else:
            raise ConfigurationError(
                f"{ds_type} specified in configuration file is not supported"
                f"implement your data class with `DatasetProcessor.register(name)"
            )

    @abstractmethod
    def process(self):
        raise NotImplementedError

    def index2entity(self, indexes):
        labels = indexes.apply(lambda id : self.id2ent[id])
        return labels

    def index_entities(self, ent: str):
        if ent not in self.ent2id:
            self.ent2id.update({ent: self.num_entities()})
            self.id2ent.update({self.num_entities()-1: ent})

        return self.ent2id[ent]

    def index_relations(self, rel: str):
        if rel not in self.rel2id:
            self.rel2id.update({rel: self.num_relations()})
            self.id2rel.update({self.num_relations()-1: rel})

        return self.rel2id[rel]

    def index_timestamps(self, ts):
        if ts not in self.ts2id:
            self.ts2id.update({ts: self.num_timestamps()})

        return self.ts2id[ts]

    def load(self):
        train_file = self.folder + "/train.txt"
        valid_file = self.folder + "/valid.txt"
        test_file = self.folder + "/test.txt"

        with open(train_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.train_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.train_raw.append(insert_line)
            else:
                self.train_raw = f.readlines()

            self.train_size = len(self.train_raw)

        with open(valid_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.valid_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.valid_raw.append(insert_line)
            else:
                self.valid_raw = f.readlines()

            self.valid_size = len(self.valid_raw)

        with open(test_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.test_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.test_raw.append(insert_line)
            else:
                self.test_raw = f.readlines()

            self.test_size = len(self.test_raw)

        # add additional files
        # -----------------------------------------------------
        # inferred, explanation, events, all
        inferred_file = self.folder + "/inferred.txt"
        explanation_file = self.folder + "/explanation.txt"
        events_file = self.folder + "/events.txt"
        all_file = self.folder + "/all.txt"

        self.inferred_raw = []
        self.explanation_raw = []
        self.events_raw = []
        self.all_raw = []

        # inferred
        with open(inferred_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.inferred_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.inferred_raw.append(insert_line)
            else:
                self.inferred_raw = f.readlines()

            self.inferred_size = len(self.inferred_raw)

        # explanation
        with open(explanation_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.explanation_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.explanation_raw.append(insert_line)
            else:
                self.explanation_raw = f.readlines()

            self.explanation_size = len(self.explanation_raw)

        # events
        with open(events_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.events_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.events_raw.append(insert_line)
            else:
                self.events_raw = f.readlines()

            self.events_size = len(self.events_raw)

        # all
        with open(all_file, "r", encoding="utf-8") as f:
            if self.reciprocal_training:
                for line in f.readlines():
                    self.all_raw.append(line)

                    insert_line = line.strip().split('\t')
                    insert_line[1] += '(RECIPROCAL)'
                    insert_line[0], insert_line[2] = insert_line[2], insert_line[0]
                    insert_line = '\t'.join(insert_line)

                    self.all_raw.append(insert_line)
            else:
                self.all_raw = f.readlines()

            self.all_size = len(self.all_raw)        

    @abstractmethod
    def process_time(self, origin: str):
        # TODO(gengyuan) use datetime
        raise NotImplementedError

    def get(self, split: str = "train"):
        # TODO(gengyuan)
        
        return {"train": self.train_set, 
            "valid": self.valid_set, 
            "test": self.test_set, 
            "pred": self.pred_set,
            "inferred": self.inferred_set,
            "explanation": self.explanation_set,
            "events": self.events_set,
            "all": self.all_set}[split]

    def num_entities(self):
        return len(self.ent2id)

    def num_relations(self):
        return len(self.rel2id)

    def num_timestamps(self):
        return len(self.ts2id)

    def num_time_identifier(self):
        return len(self.ts2id)

    def filter(self, type="static", target="o") -> Dict[str, List]:
        """
        Returns generated link prediction queries.
        Removes the specified target (either s, p or o) out of a copy of each triple respectively quadruple
        (if specified type is static respectively time-aware) and adds each answer as the last element.
        """
        self.config.assert_true(type in ["static",
                                         "time-aware",
                                         "raw"],
                                f"{type} filtering is not implemented; use static/time-aware/raw filtering.")
        self.config.assert_true(target in ["s", "p", "o"],
                                "Only support s(ubject)/p(redicate)/o(bject) prediction task")

        filtered_data = defaultdict(list)

        if type != "raw":
            all_tuples = self.all_triples if type == "static" else self.all_quadruples

            for tup in all_tuples:
                query = tup.copy()

                # TODO(gengyuan) enum
                missing = query[SPOT[target].value - 1]
                query[SPOT[target].value - 1] = None

                query_k = f"{query[0]}-{query[1]}-{query[2]}"

                if type == "time-aware":
                    query_k += f"-{query[3]}"

                filtered_data[query_k].append(missing)

        return filtered_data

    def info(self):
        self.config.log('==============================================')
        self.config.log(f'Dataset type : {self.config.get("dataset.name")}')
        self.config.log(f"Number of entities : {self.num_entities()}")
        self.config.log(f"Number of relations : {self.num_relations()}")
        self.config.log(f"Number of temporal identifiers : {self.num_timestamps()}")
        self.config.log(f"\n")
        self.config.log(f"Train set size : {len(self.train_set['triple'])}")
        self.config.log(f"Valid set size : {len(self.valid_set['triple'])}")
        self.config.log(f"Test set size : {len(self.test_set['triple'])}")
        self.config.log('==============================================')



@DatasetProcessor.register(name="ExpCPS")
class ExpCPSDatasetProcessor(DatasetProcessor):
    def process(self):
        start_day = 25
        end_day = start_day
        all_timestamp = get_all_hours_of_days(start_day,end_day)
        self.ts2id = {ts: (arrow.get(ts) - arrow.get('1973-12-'+str(start_day)+"T00:00:40")).days for ts in all_timestamp}

        for rd in self.train_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            self.train_set['triple'].append([head, rel, tail])
            self.train_set['timestamp_id'].append([ts_id])
            self.train_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            self.all_triples.append([head, rel, tail])
            self.all_quadruples.append([head, rel, tail, ts_id])

        for rd in self.valid_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.valid_set['triple'].append([head, rel, tail])
                self.valid_set['timestamp_id'].append([ts_id])
                self.valid_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

# add triples to be predicted -----------------------------------------------------------------------------------------------------
        if (self.head_target != "") & (self.rel_target != ""):
            head = self.index_entities(self.head_target)
            rel = self.index_relations(self.rel_target)
            tail = 0
            ts = self.process_time(self.ts_target)
            ts_id = self.index_timestamps(ts)

            
            self.pred_set['triple'].append([head, rel, tail])
            self.pred_set['timestamp_id'].append([ts_id])
            self.pred_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

            # self.all_triples.append([head, rel, tail])
            # self.all_quadruples.append([head, rel, tail, ts_id])

# --------------------------- -----------------------------------------------------------------------------------------------------

        for rd in self.test_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.test_set['triple'].append([head, rel, tail])
                self.test_set['timestamp_id'].append([ts_id])
                self.test_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                self.all_triples.append([head, rel, tail])
                self.all_quadruples.append([head, rel, tail, ts_id])

        # inferred
        for rd in self.inferred_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            # head = self.index_entities(head)
            # rel = self.index_relations(rel)
            # tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.inferred_set['triple'].append([head, rel, tail])
                self.inferred_set['timestamp_id'].append([ts_id])
                self.inferred_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                # self.all_triples.append([head, rel, tail])
                # self.all_quadruples.append([head, rel, tail, ts_id])

        # explanation
        for rd in self.explanation_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            # head = self.index_entities(head)
            # rel = self.index_relations(rel)
            # tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.explanation_set['triple'].append([head, rel, tail])
                self.explanation_set['timestamp_id'].append([ts_id])
                self.explanation_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                # self.all_triples.append([head, rel, tail])
                # self.all_quadruples.append([head, rel, tail, ts_id])

        # events
        for rd in self.events_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            # head = self.index_entities(head)
            # rel = self.index_relations(rel)
            # tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.events_set['triple'].append([head, rel, tail])
                self.events_set['timestamp_id'].append([ts_id])
                self.events_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                # self.all_triples.append([head, rel, tail])
                # self.all_quadruples.append([head, rel, tail, ts_id])

        # all
        for rd in self.all_raw:
            head, rel, tail, ts = rd.strip().split('\t')
            head = self.index_entities(head)
            rel = self.index_relations(rel)
            tail = self.index_entities(tail)
            ts = self.process_time(ts)
            ts_id = self.index_timestamps(ts)

            if '(RECIPROCAL)' not in rd:
                self.all_set['triple'].append([head, rel, tail])
                self.all_set['timestamp_id'].append([ts_id])
                self.all_set['timestamp_float'].append(list(map(lambda x: int(x), ts.split('-'))))

                # self.all_triples.append([head, rel, tail])
                # self.all_quadruples.append([head, rel, tail, ts_id])

    def process_time(self, origin: str):
        all_resolutions = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.config.assert_true(self.resolution in all_resolutions, f"Time granularity should be {all_resolutions}")

        ts = re.split(r"[ :\-T]+",origin) 
        ts = ts[:all_resolutions.index(self.resolution) + 1]
        ts = '-'.join(ts)

        return ts

class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, List], datatype: Optional[List[str]] = None):
        super().__init__()

        self.dataset = dataset
        self.datatype: List = datatype

        # TODO(gengyuan) assert the lengths of all lists in self.dataset
        # use self.config.assert_true(condition, message)
        # assert all( for i in dataset.items())

    def __len__(self):
        # TODO(gengyuan) calculate the length
        return len(self.dataset['triple'])

    def __getitem__(self, index, train=True):
        sample = torch.Tensor(self.dataset['triple'][index])

        for type in self.datatype:
            if type == 'timestamp_id':
                timestamp_id = torch.Tensor(self.dataset['timestamp_id'][index])
                sample = torch.cat([sample, timestamp_id], dim=0)

            elif type == 'timestamp_float':
                timestamp_float = torch.Tensor(self.dataset['timestamp_float'][index])
                sample = torch.cat([sample, timestamp_float], dim=0)
            else:
                raise NotImplementedError

        return sample
