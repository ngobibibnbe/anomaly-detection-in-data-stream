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
class class_KitNet:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(X,right,nbr_anomalies,gap, scoring_metric="merlin"):

        #@jit
        def Kitnet(X,window_size, max_size_ae):
            """
            Malheureusement le concept drift n'est pas encore implémenté dans pysad nous devons le faire manuellement
            """
            np.random.seed(61)  # Fix random seed.
            X_all =np.array(X)
            iterator = ArrayStreamer(shuffle=False)
            #print(X_all.shape)
            X_all=X_all#.reshape(1,-1)[0:2]
            # Fit reference window integration to first 100 instances initially.
            model=models.KitNet( grace_anomaly_detector =window_size,max_size_ae=max_size_ae  )#, )grace_feature_mapping=window_size,
            scores=[]
            #scores= scores+ np.zeros(len(X_all[:initial_window])).tolist()
            end=0
            for idx, x in enumerate(X_all):
                #x=X_all[idx:idx+window_size]
                """if len(x)<window_size:
                    end=len(X_all)-idx
                    break"""
                model.fit_partial(x)  # Fit to the instance.
                score = model.score_partial(x)  # Score the instance.
                #print(score)
                scores.append(score)
            
            scores =np.concatenate([np.array(scores), np.zeros(end)])
            scores =scores.squeeze()
            
            #print("****", np.argwhere(np.isnan(scores)).reshape(1,-1)[0])
            try :
                if len(np.isnan(scores).reshape(1,-1)[0])!=0:
                    scores[np.argwhere(np.isnan(scores)).reshape(1,-1)[0][0]] =0
            except:
                print("erreur bizarre")
            #print("****", np.argwhere(np.isnan(scores)).reshape(1,-1)[0])
            #print(len(X_all),"***",len(scores))
            return scores

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
            #print("****", precision, len(right))
            precision =precision/len(identified)
            #print("*****", precision, len(identified))
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
            """Y_test=np.zeros(len(X))

            for real in right:
                real=int(real)
                Y_test[real]=1

            precision, recall, threshold = metrics.precision_recall_curve(Y_test,scores)
            print("***",precision)
            f1score =np.array([2*precision[i]*recall[i]/(recall[i]+precision[i]) for i, pre in enumerate(precision)])
            
            arg = np.argmax(f1score)
            precision, recall, thresholds = metrics.precision_recall_curve(Y_test,scores)
            precision, recall, thresholds =precision[:-1], recall[:-1], thresholds[:-1]
            f1_score = 2*precision*recall/(precision + recall)

            q = list(zip(f1_score, thresholds))

            thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
            threshold=thres
            arg=np.where(thresholds==thres)
            print("*****",threshold,  f1_score[arg])
            #hold[arg]
            #print("***",arg, f1score[arg])
            #print(threshold)
            print("***",right, Y_test)"""
            
            thresholds = np.unique(scores)[:20]
            f1_scores =[]
            for threshold in thresholds:
                labels=np.where(scores<threshold,0,1)
                f1_scores.append(scoring(labels))
            
            q = list(zip(f1_scores, thresholds))

            thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
            threshold=thres
            arg=np.where(thresholds==thres)
            #print("*****",arg[0],threshold)

            """threshold=0.00001
            tmp=scores.copy()
            real_indices=np.array([0])
            real_indices=np.delete(real_indices, 0)
            while len(real_indices)<nbr_anomalies and len(tmp)!=1:
                threshold = np.amax(tmp) #max(tmp)
                indices = [i for i,val in enumerate(tmp) if val==threshold]#tmp.index(max(tmp))
                tmp=np.delete(tmp, indices)
                indices= [i for i,val in enumerate(scores) if val==threshold] 
                
                
                indices =np.where(scores == threshold)
                for indice in indices:
                    if check(indice,real_indices,gap):
                        real_indices = np.append(real_indices,indice)
                    #print("**",threshold,(real_indices))"""""""""
            return np.where(scores<threshold,0,1)# i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]




        def objective(args):
            print(args)
            
            #try:
            scores= Kitnet(X,window_size=args["window_size"],max_size_ae=args["max_size_ae"] )
            scores= score_to_label(nbr_anomalies,scores,gap) 
            
            
            return 1/(1+scoring(scores))#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_window_size =np.arange(100, len(X)/4) #2000
        possible_max_size_ae  =np.arange(1,np.array(X).shape[1]) #2000
        space2 ={ "window_size":hp.choice("window_size_index",possible_window_size),
        "max_size_ae":hp.choice("max_size_ae_index",possible_max_size_ae )}
        trials = Trials()
        
        
        best = fmin(fn=objective,space=space2, algo=tpe.rand.suggest, max_evals=30,trials = trials)
        #print("****************")
        start =time.monotonic()
        real_scores= Kitnet(X,window_size=possible_window_size[best["window_size_index"]], max_size_ae=possible_max_size_ae[best["max_size_ae_index"]] )
        end =time.monotonic()
        best_param={"window_size":possible_window_size[best["window_size_index"]], "max_size_ae":possible_max_size_ae[best["max_size_ae_index"]] }
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start


        
