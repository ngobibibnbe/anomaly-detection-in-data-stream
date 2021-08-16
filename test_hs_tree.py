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

def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        real_indice=int(real_indice)
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice-gap,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag


import math
import sys
from datetime import datetime
from score_nab import evaluating_change_point
class class_hstree:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(dataset,X,right,nbr_anomalies,gap, scoring_metric="merlin"):

        #@jit
        def HStree(X, initial_window, window_size, num_trees, max_depth):
            """
            Malheureusement le concept drift n'est pas encore implémenté dans pysad nous devons le faire manuellement
            """
            initial_window=window_size
            
            np.random.seed(61)  # Fix random seed.
            X_all =np.array(X)
            iterator = ArrayStreamer(shuffle=False)
            A=X_all#.reshape(-1,1)
            #print()
            # Fit reference window integration to first 100 instances initially.
            model=models.HalfSpaceTrees(feature_mins=np.array(A.min(axis=0)), feature_maxes=np.array(A.max(axis=0)), window_size=window_size, num_trees=num_trees, max_depth=max_depth, initial_window_X=X[:initial_window])
            scores=[]
            #scores= scores+ np.zeros(len(X_all[:initial_window])).tolist()
            
            for x in tqdm(iterator.iter(X_all)):
                model.fit_partial(x)  # Fit to the instance.
                score = model.score_partial(x)  # Score the instance.
                #print(score)
                scores.append(score)
            #print(scores)
            return scores

                        
        #right=[387,948,1485]
        #nbr_anomalies=3
        
        def scoring(scores):
            score=0
            for real in right:
                real=int(real)
                if 1 in scores[real-gap:real+gap]:
                    score+=1
            recall=score/nbr_anomalies #recall

            precision=0
            identified =np.where(scores==1)
            identified=identified[0]
            for found in identified:
                if not check(found,right,gap):
                    precision+=1
            precision =precision/len(identified)
            try :
                score =2*(recall*precision)/(recall+precision) 
            except :
                score=0 
            if scoring_metric=="nab":
                real_label = [int(0) for i in X]
                for element in right:
                    real_label[int(element)]=int(1)
                    real_label_frame=pd.DataFrame(real_label, columns=['changepoint']) 
                    scores_frame=pd.DataFrame(scores, columns=['changepoint']) 
                    real_label_frame["datetime"] =pd.to_datetime(real_label_frame.index, unit='s')
                    scores_frame["datetime"] =pd.to_datetime(scores_frame.index, unit='s')
                    real_label_frame =real_label_frame.set_index('datetime')
                    scores_frame =scores_frame.set_index('datetime')                
                nab_score=evaluating_change_point([real_label_frame.changepoint],[scores_frame.changepoint]) 
                nab_score=nab_score["Standart"]  
                score=nab_score   
            return score
        
        def score_to_label(nbr_anomalies,scores,gap):
            
            thresholds = np.unique(scores)
            f1_scores =[]
            for threshold in thresholds:
                labels=np.where(scores<threshold,0,1)
                f1_scores.append(scoring(labels))
            
            q = list(zip(f1_scores, thresholds))

            thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
            threshold=thres
            arg=np.where(thresholds==thres)
           
            return np.where(scores<threshold,0,1)# i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]

        def objective(args):
            print(args)
            
            #try:
            scores= HStree(X,initial_window=args["initial_window"],window_size=args["window_size"],
            num_trees=args["num_trees"], max_depth=args["max_depth"]   )
            scores= score_to_label(nbr_anomalies,scores,gap) 
            
            
            return 1/(1+scoring(scores))#{'loss': 1/1+score, 'status': STATUS_OK}


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
        
        
        best = fmin(fn=objective,space=space2, algo=tpe.rand.suggest, max_evals=30,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= HStree(X,initial_window=possible_initial_window[best["initial_window_index"]],window_size=possible_window_size[best["window_size_index"]],
        num_trees=possible_nbr_tree [best["num_trees"]], max_depth=possible_max_depth [best["max_depth"]] )
        end =time.monotonic()
        best_param={"initial_window":possible_initial_window[best["initial_window_index"]],"window_size":possible_window_size[best["window_size_index"]] , 
        "num_trees":possible_nbr_tree [best["num_trees"]], "max_depth":possible_max_depth [best["max_depth"]]}

        """except :
            print("there was an error")
            best_param={"initial_window":"RAS","window_size":"RAS","Bucket_index":"RAS" }
        """
        #real_scores=np.zeros(len(X))
        #iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        #print("the final score is", scoring(scores_label),identified)
        """if scoring_metric=="nab":
            real_label = np.zeros(len(X))
            for element in right:
                real_label[int(element)]=1
            real_label_frame=pd.DataFrame(real_label, columns=['changepoint']) 
            scores_frame=pd.DataFrame(scores_label, columns=['changepoint']) 
            real_label_frame["datetime"] =pd.to_datetime(real_label_frame.index, unit='s')
            scores_frame["datetime"] =pd.to_datetime(scores_frame.index, unit='s')
            real_label_frame =real_label_frame.set_index('datetime')
            scores_frame =scores_frame.set_index('datetime')
        
            nab_score=evaluating_change_point([real_label_frame.changepoint],[scores_frame.changepoint]) 
            nab_score=nab_score["Standart"]  
            return real_scores, scores_label, identified,nab_score, best_param, end-start"""    
        return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start


        

