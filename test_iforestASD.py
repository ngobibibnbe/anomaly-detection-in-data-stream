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
from score_nab import evaluating_change_point
# methode avec matrix profile


def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        real_indice=int(real_indice)
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice-gap,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag


def overlapping_merlin(identified, expected,gap):
  score=0
  for expect in expected: 
    if identified in [*range (int(expect)-gap,int(expect)+gap)]:
      score+=1
  return score/len(expected)

class class_iforestASD:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
            
    def test(X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        def iforestASD(X,window_size,n_estimators, max_features):
            X =np.array(X)
            """
            Malheureusement le concept drift n'est pas encore implémenté dans pysad nous devons le faire manuellement
            n_estimators est le nombre d'arbres nécéssaire
            Window_size est la taille de la fenêtre
            """
            initial=window_size
            np.random.seed(61)  # Fix random seed.
            drift_detector = drift.ADWIN()
            X_all =X
            #iterator = ArrayStreamer(shuffle=False)
            model=models.IForestASD(initial_window_X=X_all[:initial],window_size=window_size)
            model.n_estimators=n_estimators
            model.max_features = max_features
            model.fit(X_all[:window_size])
            

            i=0
            scores=np.array([])
            batchs=range (0,len(X_all),window_size)
            flag=False
            for batch in range (0,len(X_all),window_size):
                i+=1
                #score =model.score(X_all[batch:batch+window_size])
                
                #print(scores)
                score =model.score(X_all[batch:batch+window_size])
                for id,one_score in enumerate(score) :
                    in_drift, in_warning =  drift_detector.update(one_score)
                    if in_drift:
                        print("changed detected at", id+batch)
                        flag=True
                        break
                if flag==True:
                    model=models.IForestASD(initial_window_X=X_all[batch:batch+window_size],window_size=window_size)
                    model.fit(X_all[batch:batch+window_size])
                    flag=False
                score =np.array(score)
                scores=np.concatenate((scores, score)) # scores+score
            #scores =np.zeros(len(X_all))#np.array(model.score(X_all))
            time.sleep(1)
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
            scores= iforestASD(X,window_size=args["window_size"],n_estimators=args["n_estimators"],max_features=args["max_features"]                                )
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_nbr_tree =np.arange(15,50)#[*range(1,100)]
        possible_window_size =np.arange(200,max(201,int(len(X)/4))) #[*range(200,1000)]
        possible_features =np.array([1,1])
        #possible_features =np.arange(np.array(X).shape[1]-1,np.array(X).shape[1])
        if np.array(X).shape[1]>1:
            possible_features =np.arange(1, np.array(X).shape[1]) #[*range(200,1000)]
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)
        , "n_estimators":hp.choice("n_estimators_index",possible_nbr_tree) , "max_features":hp.choice("max_features",possible_features)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.rand.suggest, max_evals=30)
        #print(best)
        start =time.monotonic()
        real_scores= iforestASD(X,max_features= possible_features[best["max_features"]],  window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        end =time.monotonic()
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"max_features":possible_features[best["max_features"]],"window_size":possible_window_size[best["window_size_index"]],"n_estimator": possible_nbr_tree[best["n_estimators_index"]]}
        #print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start

