# Import modules.
import numpy as np
#from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pysad.utils import Data
import scipy.io
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


#from test_1 import class_iforestASD
# methode avec matrix profile
def plot_time_series(df, title=None, ano=None, ano_name='None'):
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=df))
	if ano!=None:
		fig.add_trace(go.Scatter(y=df[ano], x=ano, name=ano_name))
	if title:
		fig.update_layout(title=title)
	return fig

def plot_fig (df, title, ):
  plt.figure(figsize=(15, 7))
  #ax = plt.plot(df.index.values, mp_adjusted)
  ax = plt.plot(df.index.values, df.values)
  plt.title(title)
  plt.show()

#@jit(nopython=True)

def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag

#@jit(nopython=True)
@register_jitable
def score_to_label(nbr_anomalies,scores,gap):
  #"""abnormal points has the right to produce various anomaly  in the same """
  
  threshold=0.00001
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
        #print("**",threshold,(real_indices))
  return np.where(scores<threshold,0,1)# [0 if i<threshold else 1 for i in scores ]


#ok



import math
import sys
from datetime import datetime
sys.path.append('MILOF/lib')
from MiLOF import MILOF

from stream_discord import class_our
from test_LAMP import class_LAMP
from test_hs_tree import class_hstree
from test_iforestASD import class_iforestASD
from score_nab import evaluating_change_point
from test_ARIMAFD import class_ARIMAFD
from test_KitNet import class_KitNet

class class_MILOF:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        def MILOF_(X,NumK,KPar,Bucket ):
            """
            LOF ne tient pas en compte les concentrations de points identiques sinon on tombe à un problème d'infini
            """
            input_file = 'MILOF/lib/test/testdata/testdata'+dataset.replace("/","_")+'.mat'
            
                
            scipy.io.savemat(input_file, mdict={'DataStream': np.array(X)})
            parameters ="""[Parser]
            InputBPFile = testdata/tau-metrics-cached-validated/tau-metrics.bp
            ProvDBPath = testdata/ProvDB.pkl
            QueueSize = 40000
            InterestFuncNum = 10

            [Analyzer]
            Width = 0
            """
            #Dimension = 1
            
            param =  "\n"+"Bucket="+str(Bucket)
            parameters+="NumK="+str(NumK)
            parameters+="\n"+"Kpar="+str(KPar)+"\nInputMatFile="+input_file+"\nDimension="+str(np.array(X).shape[1])
            parameters+=param
            #print(parameters)
            file_name =dataset.replace("/","_")
            text_file = open("MILOF/conf/"+file_name+".cfg", "w")
            n = text_file.write(parameters)
            text_file.close()

            scores,X_unique = MILOF("MILOF/conf/"+file_name+".cfg")
            #print("*",len(X))
            real_scores=[0 for i in X]
            scores =[1/i for i in scores]
            maxi=max(scores)
            
            for idx, x in enumerate(X):
                indices = [i for i, a in enumerate(np.array(X_unique) ) if (x == a).all()]
                real_scores[idx]=scores[indices[0]]/maxi
            #% cd /content/drive/MyDrive/Projets/stage
            return real_scores 

                        
        #right=[387,948,1485]
        #nbr_anomalies=3
        
        def scoring(scores):
            score=0
            for real in right:
                real=int(real)
                if 1 in scores[real-gap:real+gap]:
                    score+=1
            score=score/nbr_anomalies
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
                return nab_score
            return score
        
        def objective(args):
            print(args)
            
            #try:
            scores= MILOF_(X,NumK=args["NumK"],KPar=args["KPar"],Bucket=args["Bucket"] )
            """except :
                print("there was an error so")
                scores=np.zeros(len(X))"""
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_NumK=np.arange(5,min(15,int(len(np.unique(np.array(X)))/2)))#[*range(1,100)]
        possible_KPar =np.arange(3,min(15,int(len(np.unique(np.array(X)))/3)) ) #[*range(200,1000)]
        possible_Bucket =np.array([min(500,len(np.unique(np.array(X)))) ])
        space2 ={"NumK":hp.choice("NumK_index",possible_NumK)
        , "KPar":hp.choice("KPar_index",possible_KPar), "Bucket":hp.choice("Bucket_index",possible_Bucket) }
        trials = Trials()
        
        #try:
        if possible_Bucket[0]<3:
            real_scores =np.zeros(len(X))
            best_param={"Numk":"RAS","KPar":"RAS","Bucket_index":"RAS" }
            end=start =time.monotonic()
        else:
            best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=30,trials = trials)
            #print(best)
            start =time.monotonic()
            real_scores= MILOF_(X,NumK=possible_NumK[best["NumK_index"]],KPar=possible_KPar[best["KPar_index"]],Bucket=possible_Bucket[best["Bucket_index"]] )
            end =time.monotonic()
            best_param={"Numk":possible_NumK[best["NumK_index"]],"KPar":possible_KPar[best["KPar_index"]],"Bucket_index":possible_Bucket[best["Bucket_index"]] }

        """except :
            print("there was an error")
            best_param={"Numk":"RAS","KPar":"RAS","Bucket_index":"RAS" }
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


