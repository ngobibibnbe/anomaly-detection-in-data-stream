# Import modules.
from sklearn.utils import shuffle
#from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.utils import Data
from tqdm import tqdm
import numpy as np

from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pysad import models
from pyod.models.iforest import IForest

from pyod.models.iforest import IForest
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import scipy.io
from river import drift
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import time
import os 
from hyperopt import fmin, tpe,hp, STATUS_OK, Trials,rand
from numba import jit, cuda
import code
#code.interact(local=locals)
import time
import os
import numba  # We added these two lines for a 500x speedup
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
actual_dataset=[0]
from base_model import BaseModel
from score_nab import evaluating_change_point
class class_hstree(BaseModel):

    def test(self,dataset,X,right,nbr_anomalies,gap, scoring_metric="merlin"):

        #@jit
        def HStree(X, initial_window, window_size, num_trees, max_depth):

            initial_window=window_size
            np.random.seed(61)  # Fix random seed.
            X_all =np.array(X)
            iterator = ArrayStreamer(shuffle=False)
            A=X_all#.reshape(-1,1)
            # Fit reference window integration to first 100 instances initially.
            model=models.HalfSpaceTrees(feature_mins=np.array(A.min(axis=0)), feature_maxes=np.array(A.max(axis=0)), window_size=window_size, num_trees=num_trees, max_depth=max_depth, initial_window_X=X[:initial_window])
            scores=[]            
            for x in tqdm(iterator.iter(X_all)):
                model.fit_partial(x)  # Fit to the instance.
                score = model.score_partial(x)  # Score the instance.
                scores.append(score)
            return scores

        def objective(args):
            scores= HStree(X,initial_window=args["initial_window"],window_size=args["window_size"],
            num_trees=args["num_trees"], max_depth=args["max_depth"]   )
            scores= self.score_to_label(scores)
            return 1/(1+self.scoring(scores))#{'loss': 1/1+score, 'status': STATUS_OK}

        possible_initial_window=np.arange(100,int(len(X)/4))#[*range(1,100)]
        possible_window_size =np.arange(200, max(201,int(len(X)/4)) ) #[*range(200,1000)]
        possible_nbr_tree =np.arange(15,35)#[*range(1,100)]  num_trees=25, max_depth=15
        possible_max_depth= np.arange(10,25)
        space2 ={"initial_window":hp.choice("initial_window_index",possible_initial_window)
        , "window_size":hp.choice("window_size_index",possible_window_size), 
         "num_trees":hp.choice("num_trees",possible_nbr_tree), 
         "max_depth":hp.choice("max_depth",possible_max_depth), 
         }
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=rand.suggest, max_evals=1,trials = trials)
        start =time.monotonic()
        real_scores= HStree(X,initial_window=possible_initial_window[best["initial_window_index"]],window_size=possible_window_size[best["window_size_index"]],
        num_trees=possible_nbr_tree [best["num_trees"]], max_depth=possible_max_depth [best["max_depth"]] )
        end =time.monotonic()
        best_param={"initial_window":possible_initial_window[best["initial_window_index"]],"window_size":possible_window_size[best["window_size_index"]] , 
        "num_trees":possible_nbr_tree [best["num_trees"]], "max_depth":possible_max_depth [best["max_depth"]]}
        scores_label =self.score_to_label(real_scores)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        return real_scores, scores_label, identified,self.scoring(scores_label), best_param, end-start

