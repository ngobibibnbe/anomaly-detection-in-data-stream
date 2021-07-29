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
#@register_jitable
def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag

#@jit(nopython=True)
#@register_jitable
def score_to_label(nbr_anomalies,scores,gap):
  #"""abnormal points has the right to produce various anomaly  in the same """
  tmp=scores.copy()
  real_indices=np.array([0])
  real_indices=np.delete(real_indices, 0)
  while len(real_indices)<nbr_anomalies and len(tmp)!=1:
    threshold = np.amax(tmp) #max(tmp)
    indices = [i for i,val in enumerate(tmp) if val==threshold]#tmp.index(max(tmp))
    tmp=np.delete(tmp, indices)
    indices= [i for i,val in enumerate(scores) if val==threshold] 
    for indice in indices:
        if check(indice,real_indices,gap):
            real_indices = np.append(real_indices,indice)
        #print("**",threshold,(real_indices))
  return np.where(scores<threshold,0,1)# [0 if i<threshold else 1 for i in scores ]


#ok

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
        def iforestASD(X,window_size,n_estimators=100):
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
            model.fit(X_all[:window_size])
            model.n_estimators=25

            i=0
            scores=np.array([])
            batchs=range (0,len(X_all),window_size)
            flag=False
            for batch in range (0,len(X_all),window_size):
                i+=1
                #score =model.score(X_all[batch:batch+window_size])
                if flag==True:
                    model=models.IForestASD(initial_window_X=X_all[batch:batch+window_size],window_size=window_size)
                    model.fit(X_all[batch:batch+window_size])
                    flag=False
                #print(scores)
                score =model.score(X_all[batch:batch+window_size])
                for id,one_score in enumerate(score) :
                    in_drift, in_warning =  drift_detector.update(one_score)
                    if in_drift:
                        print("changed detected at", id+batch)
                        flag=True
                        break
                
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
            scores= iforestASD(X,window_size=args["window_size"],n_estimators=args["n_estimators"]
                                )
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_nbr_tree =np.arange(15,35)#[*range(1,100)]
        possible_window_size =np.arange(200,max(201,int(len(X)/4))) #[*range(200,1000)]
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)
        , "n_estimators":hp.choice("n_estimators_index",possible_nbr_tree)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=30)
        #print(best)
        start =time.monotonic()
        real_scores= iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        end =time.monotonic()
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"window_size":possible_window_size[best["window_size_index"]],"n_estimator": possible_nbr_tree[best["n_estimators_index"]]}
        #print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start




