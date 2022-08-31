#!/usr/bin/env python
# coding: utf-8

# # 2. Build Models
# 
# Use preprocessed data from notebook **01_Preprocessing.ipynb** and train KGE models on the data.

# In[ ]:


#! /usr/venv/thesis/bin/python3


# In[ ]:


# %pip install -r ../requirements.txt 


# In[ ]:


from helper import *


# ## Import Data

# In[ ]:


training = get_triples('../data/splits/trainingNT.pkl')
testing = get_triples('../data/splits/testNT.pkl')
validation = get_triples('../data/splits/test_validNT.pkl')

MODEL_NAME = "TransH"


# In[ ]:


"""An implementation of TransH.
There are errors caused when using the pykeen implementation of TransH, so I changed the implementation of the regularizer to run smoothly"""

import itertools
from typing import Any, ClassVar, Mapping, Type

from class_resolver import HintOrType, OptionalKwargs
from torch.nn import functional, init

from pykeen.models.nbase import ERModel
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn import TransHInteraction
from pykeen.regularizers import NormLimitRegularizer, OrthogonalityRegularizer, Regularizer
from pykeen.typing import Hint, Initializer, Constrainer

__all__ = [
    "TransH",
]


class MyTransH(ERModel):
    r"""An implementation of TransH [wang2014]_.

    This model extends :class:`pykeen.models.TransE` by applying the translation from head to tail entity in a
    relational-specific hyperplane in order to address its inability to model one-to-many, many-to-one, and
    many-to-many relations.

    In TransH, each relation is represented by a hyperplane, or more specifically a normal vector of this hyperplane
    $\textbf{w}_{r} \in \mathbb{R}^d$ and a vector $\textbf{d}_{r} \in \mathbb{R}^d$ that lies in the hyperplane.
    To compute the plausibility of a triple $(h,r,t)\in \mathbb{K}$, the head embedding $\textbf{e}_h \in \mathbb{R}^d$
    and the tail embedding $\textbf{e}_t \in \mathbb{R}^d$ are first projected onto the relation-specific hyperplane:

    .. math::

        \textbf{e'}_{h,r} = \textbf{e}_h - \textbf{w}_{r}^\top \textbf{e}_h \textbf{w}_r

        \textbf{e'}_{t,r} = \textbf{e}_t - \textbf{w}_{r}^\top \textbf{e}_t \textbf{w}_r

    where $\textbf{h}, \textbf{t} \in \mathbb{R}^d$. Then, the projected embeddings are used to compute the score
    for the triple $(h,r,t)$:

    .. math::

        f(h, r, t) = -\|\textbf{e'}_{h,r} + \textbf{d}_r - \textbf{e'}_{t,r}\|_{p}^2

    .. seealso::

       - OpenKE `implementation of TransH <https://github.com/thunlp/OpenKE/blob/master/models/TransH.py>`_
    ---
    citation:
        author: Wang
        year: 2014
        link: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )
    # #: The custom regularizer used by [wang2014]_ for TransH
    # regularizer_default: ClassVar[Type[Regularizer]] = NormLimitRegularizer
    # #: The settings used by [wang2014]_ for TransH
    # # The regularization in TransH enforces the defined soft constraints that should computed only for every batch.
    # # Therefore, apply_only_once is always set to True.
    # regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
    #     weight=0.05, apply_only_once=True, dim=-1, p=2, power_norm=True, max_norm=1.0
    # )
    # #: The custom regularizer used by [wang2014]_ for TransH
    # relation_regularizer_default: ClassVar[Type[Regularizer]] = OrthogonalityRegularizer
    # #: The settings used by [wang2014]_ for TransH
    # relation_regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
    #     weight=0.05, apply_only_once=True, epsilon=1e-5
    # )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 2,
        entity_initializer: Hint[Initializer] = init.xavier_normal_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        # entity_regularizer: HintOrType[Regularizer] = None,
        # entity_regularizer_kwargs: OptionalKwargs = None,
        relation_initializer: Hint[Initializer] = init.xavier_normal_,
        relation_constrainer: Hint[Constrainer] = functional.normalize,
        # relation_regularizer: HintOrType[Regularizer] = None,
        # relation_regularizer_kwargs: OptionalKwargs = None,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransH.

        :param embedding_dim:
            the entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param scoring_fct_norm:
            the :math:`l_p` norm applied in the interaction function. Is usually ``1`` or ``2.``.

        :param entity_initializer:
            the entity initializer function
        :param entity_regularizer:
            the entity regularizer. Defaults to :attr:`pykeen.models.TransH.regularizer_default`
        :param entity_regularizer_kwargs:
            keyword-based parameters for the entity regularizer. If `entity_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.regularizer_default_kwargs` will be used instead

        :param relation_initializer:
            relation initializer function
        :param relation_regularizer:
            the relation regularizer. Defaults to :attr:`pykeen.models.TransH.relation_regularizer_default`
        :param relation_regularizer_kwargs:
            keyword-based parameters for the relation regularizer. If `relation_regularizer` is None,
            the default from :attr:`pykeen.models.TransH.relation_regularizer_default_kwargs` will be used instead

        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.ERModel`
        """
        super().__init__(
            interaction=TransHInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations_kwargs=[
                # translation vector in hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    constrainer=relation_constrainer,
                    regularizer=regularizer,
                    regularizer_kwargs=regularizer_kwargs,
                    
                ),
                # normal vector of hyperplane
                dict(
                    shape=embedding_dim,
                    initializer=relation_initializer,
                    # normalise the normal vectors to unit l2 length
                    constrainer=relation_constrainer,
                    regularizer=regularizer,
                    regularizer_kwargs=regularizer_kwargs,
                ),
            ],
            **kwargs,
        )


# ## Building Embedding Pipeline with default values, no HPO
# 
# try to put transE with Adam optimizer in a pipeline  - no hpo for now
# 
# ### TransH, Adam Optimizer

# In[ ]:


# TransE, Adam Optimizer, no HPO
OPTIMIZER_NAME = 'Adam'
NUM_EPOCHS = 100


# In[ ]:


result = pipeline(
    training = training,
    validation = validation,
    testing = testing,
    model = MyTransH,
    optimizer = OPTIMIZER_NAME,
    training_kwargs = dict(
        num_epochs = NUM_EPOCHS,
        checkpoint_name = 'TransH_default.pt',
        checkpoint_directory = "../checkpoints/",
        checkpoint_frequency = 30,
    ),
    stopper = 'early',
    random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + 'TransH_default')
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + 'TransH_default' + ".csv")

print("-------------------\n Successfully trained "+ 'TransH_default'+ "\n-------------------")

torch.cuda.empty_cache() 


# ## Use HPO for Model Training
# 
# ### TransH
# 

# In[ ]:


from pykeen.hpo import hpo_pipeline
from optuna.samplers import GridSampler

# Transe, Adam Optimizer, HPO
result = hpo_pipeline(
    n_trials = 30,
    training = training,
    validation = validation,
    testing = testing,
    model = MyTransH,
    # regularizer_kwargs = dict(weight = 1.0),
    sampler=GridSampler,
    sampler_kwargs=dict(
        search_space={
            "model.embedding_dim": [32, 64, 128],
            "model.scoring_fct_norm": [2],
            "loss": 'marginranking',
            "loss.margin": [1.0,2.0],
            "optimizer.lr": [1.0e-03],
            "negative_sampler":'basic',
            "negative_sampler.num_negs_per_pos": [32],
            "training.num_epochs": [100],
            "training.batch_size": [128],
            "stopper":'early',
            "optimizer":["Adam", "SGD"], 
        },
    ),
    
#    optimizer = OPTIMIZER_NAME,
    save_model_directory="../checkpoints/" + MODEL_NAME + '_hpo/',
    stopper = 'early',
  #  random_seed = 1503964,
)

# save the model
result.save_to_directory('../models/' + MODEL_NAME + '_hpo')

print("-------------------\n Successfully trained" + MODEL_NAME + '_hpo'+ "\n-------------------")

torch.cuda.empty_cache() 


# In[ ]:


import json
file = '../models/' + MODEL_NAME + '_hpo/best_pipeline/pipeline_config.json'

with open(file) as json_file:
    config = json.load(json_file)
    
config["pipeline"]["training"] = training
config["pipeline"]["testing"] = testing
config["pipeline"]["validation"] = validation
config["pipeline"]["model"] = "transh"


# In[ ]:


from pykeen.pipeline import pipeline_from_config

result = pipeline_from_config(config)
result.save_to_directory('../models/' + MODEL_NAME + '_hpo' + "bestModel")
metrics = result.metric_results.to_df()
metrics.to_csv("../model_results/" + MODEL_NAME + '_hpo' + "bestModel" + ".csv")

print("-------------------\n Successfully evaluated " + MODEL_NAME + '_hpo'+ "\n-------------------")

torch.cuda.empty_cache() 

