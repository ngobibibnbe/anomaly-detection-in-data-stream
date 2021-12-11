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



class BaseModel():
    """This class represent te model Online ARIMA 
    
    It test the method on a dataset, but also perform bayesian hyperparameter optimization over 30 iteration to select the best hyperparameters

    """

    def scoring(self, scores):
        
        """provide either the merlin score or the nab score of the method on the dataset

        for evaluating with the NAB score we are using the evaluating_change_point function of NAB (taken on the SKAB github)

        :param scores: labels provided by the method on each instance (0 or 1 )
        :type scores: List
        :return: the score of the method on the dataset, either the merlin score or the NAB one 
        :rtype: float
        """
        
        def check (indice, real_indices,gap ):
            """This function check if an indice (indice) is a self.gap near to other indices (real_indices): indice ia at list in one of  [one_real_indice-self.gap, one_real_indice+self.gap] for one_real_indice in real_indices

            This function is usefull when we are processing the precision of the f1-score with the MERLIN approach 

            :param indice: the indice to test
            :type indice: int
            :param real_indices: the set of indices with which we test
            :type real_indices: List of int
            :param self.gap: the self.gap
            :type self.gap: int
            :return: boolean saying if yes or no it's near from one of  real_indices
            :rtype: boolean
            """
            Flag=True
            for real_indice in real_indices:
                real_indice=int(real_indice)
                #print(indice, [*range(real_indice-self.gap,real_indice+self.gap)])
                search = np.arange(real_indice-gap,real_indice+gap)
                if indice in search:
                    Flag=False
            return Flag


        if self.scoring_metric=="merlin":
            score=0
            for real in self.right:
                """in this loop we check if the method find an anomaly 1% around the real position of each anomaly, this to have the recall
                """
                real=int(real)
                if 1 in scores[real-self.gap:real+self.gap]:
                    score+=1
            recall=score/self.nbr_anomalies #recall

            precision=0
            identified =np.where(scores==1)
            identified=identified[0]
            for found in identified:
                if not check(found,self.right,self.gap):
                    precision+=1
            precision =precision/len(identified) # la methode est precise si elle arrive à plus trouveer de zones anormales que non normales 
            try :
                score =2*(recall*precision)/(recall+precision) 
            except :
                score=0  

        if self.scoring_metric=="nab":
            
            real_label = [int(0) for i in self.X]
            for element in self.right:
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
        
    def score_to_label(self,scores):
        """Since the method retrurns anomalies score, we should found the thresholds which allow the method to separate score representing anomalies than those not enough high to be an anomaly

        The threshold is Fixed as in the NAB benchmark done on the HTM paper of 2017, we research the best threshold among possible scores 
        The best threshold is the one providing the best score to the method on the dataset.
        In this function we test all scores for each threshold and we select the threshold that provide the best score

        
        :param scores: scores output by the method to each data instance of the stream
        :type scores: np array
        :return: return a list of boolean with 0 if the instance is normal and 1 if not. The assignement of label is the one that provided the best score
        :rtype: List
        """
        
        thresholds = np.unique(scores)
        f1_scores =[]
        for threshold in thresholds:
            labels=np.where(scores<threshold,0,1)
            f1_scores.append(self.scoring(labels))
        
        q = list(zip(f1_scores, thresholds))

        thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
        threshold=thres
        arg=np.where(thresholds==thres)
        
        return np.where(scores<threshold,0,1)# i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]


