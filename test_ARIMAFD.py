# Import modules.
from sklearn.utils import shuffle
#from pysad.evaluation import AUROCMetric
import numpy as np
from sklearn.utils import shuffle

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
from arimafd import *
from base_model import BaseModel
class class_ARIMAFD(BaseModel):
    """This class represent te model Online ARIMA 
    
    It test the method on a dataset, but also perform bayesian hyperparameter optimization over 30 iteration to select the best hyperparameters

    """

    def test(self,df,X,right,nbr_anomalies,gap,scoring_metric="merlin"):
        """Insure the full test process
        
        -for each hyperparameter 
            -test the method and record the score
        -keep best hyperparameters and the score, the execution time on each best hyperparameters


        :param df: the Dataframe containing the data
        :type df: Data
        :param X: Data in an array
        :type X: np array
        :param right: List of real anomaly position
        :type right: List of int
        :param nbr_anomalies: number of anomalies
        :type nbr_anomalies: int
        :param gap: the gap
        :type gap: int
        :param scoring_metric: the scoring metric either nab or MERLIN, defaults to "merlin"
        :type scoring_metric: str, optional
        :return:  real_scores(score of the best method), scores_label(labels of instances with the best method), identified (List of anomaly identified),self.scoring(scores_label), best_param (the best hyperparameters), end-start (execution time of the method)
        :rtype: [type]
        """

        #@jit
        def ARIMAFD(X,AR_window_size, d):
            X =np.array(X)
            X=X#.reshape(-1,1)
            X=pd.DataFrame(X,columns=df.columns)
            for i in range(d):
                X=X.diff()
            X=X.iloc[d:]
            a =anomaly_detection(X)
            a.tensor =a.generate_tensor(ar_order=AR_window_size)
            a.proc_tensor(No_metric=1)
            scores=np.concatenate([np.zeros(AR_window_size),a.generate_tensor(ar_order=AR_window_size)[:,0,0].squeeze()])# a.bin_metric]) # Joining arr1 and arr2
            scores=np.concatenate([scores,np.zeros(d)])# a.bin_metric]) # Joining arr1 and arr2
            return scores

        def objective(args):
            """This function is in charge of testing the method 

            :param args: Dictionnary of hyperparameters
            :type args: Dictionnary
            :return: the loss = the inverse of the (score+1) of the method with those hyperparameters. 
            :rtype: float
            """
            print(args)
            scores= ARIMAFD(X,AR_window_size=args["AR_window_size"],d=args["d"])
            scores =self.score_to_label(scores)
            
            return 1/(1+self.scoring(scores))#self.scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}

        #possible_nbr_tree =np.arange(1,25)#[*range(1,100)]
        possible_AR_window_size =np.arange(0, 100) #[*range(200,1000)]
        possible_d =np.arange(0,10)
        space2 ={"AR_window_size":hp.choice("AR_window_size_index",possible_AR_window_size), "d":hp.choice("d_index",possible_d) }
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.rand.suggest, max_evals=1,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= ARIMAFD(X,AR_window_size=possible_AR_window_size[best["AR_window_size_index"]],d=possible_d[best["d_index"]])
        end =time.monotonic()
        scores_label =self.score_to_label(real_scores)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"AR_window_size":possible_AR_window_size[best["AR_window_size_index"]] , "differencing-order":possible_d[best["d_index"]]}
        #print("the final score is", self.scoring(scores_label),identified)
        return real_scores, scores_label, identified,self.scoring(scores_label), best_param, end-start




