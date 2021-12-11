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
from sklearn.utils import shuffle
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
from hyperopt import fmin, tpe,hp, STATUS_OK, Trials
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
from pysad.models import KitNet
from sklearn import metrics
from base_model import BaseModel


import math
import sys
from datetime import datetime
from score_nab import evaluating_change_point
class class_KitNet(BaseModel):
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(self,X,right,nbr_anomalies,gap, scoring_metric="merlin"):
        self.scoring_metric =scoring_metric
        self.right =right
        self.gap =gap
        self.nbr_anomalies= nbr_anomalies
        self.X=X

        #@jit
        def Kitnet(X,window_size, max_size_ae):
            np.random.seed(61)  # Fix random seed.
            X_all =np.array(X)
            X_all=X_all#.reshape(1,-1)[0:2]
            # Fit reference window integration to first 100 instances initially.
            model=models.KitNet( grace_anomaly_detector =window_size,max_size_ae=max_size_ae  )#, )grace_feature_mapping=window_size,
            scores=[]
            end=0
            for idx, x in enumerate(X_all):
                #x=X_all[idx:idx+window_size]
                """if len(x)<window_size:
                    end=len(X_all)-idx
                    break"""
                model.fit_partial(x)  # Fit to the instance.
                score = model.score_partial(x)  # Score the instance.
                scores.append(score)
            
            scores =np.concatenate([np.array(scores), np.zeros(end)])
            scores =scores.squeeze()
            
            try :
                if len(np.isnan(scores).reshape(1,-1)[0])!=0:
                    scores[np.argwhere(np.isnan(scores)).reshape(1,-1)[0][0]] =0
            except:
                print("erreur bizarre")
            return scores

       
        
        def objective(args):
           
            #try:
            scores= Kitnet(X,window_size=args["window_size"],max_size_ae=args["max_size_ae"] )
            scores= self.score_to_label(scores)             
            return 1/(1+self.scoring(scores))#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_window_size =np.arange(100, len(X)/4) #2000
        possible_max_size_ae  =np.arange(1,np.array(X).shape[1]) #2000
        space2 ={ "window_size":hp.choice("window_size_index",possible_window_size),
        "max_size_ae":hp.choice("max_size_ae_index",possible_max_size_ae )}
        trials = Trials()
        
        
        best = fmin(fn=objective,space=space2, algo=tpe.rand.suggest, max_evals=1,trials = trials)
        #print("****************")
        start =time.monotonic()
        real_scores= Kitnet(X,window_size=possible_window_size[best["window_size_index"]], max_size_ae=possible_max_size_ae[best["max_size_ae_index"]] )
        end =time.monotonic()
        best_param={"window_size":possible_window_size[best["window_size_index"]], "max_size_ae":possible_max_size_ae[best["max_size_ae_index"]] }
        scores_label =self.score_to_label(real_scores)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        return real_scores, scores_label, identified,self.scoring(scores_label), best_param, end-start


        
