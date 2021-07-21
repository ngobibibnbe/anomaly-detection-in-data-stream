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
@register_jitable
def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice-gap,real_indice+gap)
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
  while len(real_indices)<nbr_anomalies:
    threshold = np.amax(tmp) #max(tmp)
    #index = tmp.index(max(tmp))
    indices= [i for i,val in enumerate(scores) if val==threshold] 
    tmp=np.delete(tmp, indices)
    
    indices =np.where(scores == threshold)
    for indice in indices:
        if check(indice,real_indices,gap):
            real_indices = np.append(real_indices,indice)
        #print("**",threshold,(real_indices))
  return np.where(scores<threshold,0,1)# [0 if i<threshold else 1 for i in scores ]


def MILOF(X):
  """
  LOF ne tient pas en compte les concentrations de points identiques sinon on tombe à un problème d'infini
  """
  import math
  % cd MILOF/lib
  from MiLOF import MILOF
  % cd test
  scipy.io.savemat('testdata/testdata.mat', mdict={'DataStream': np.array(X)})
  parameters= """[Parser] 
  InputBPFile = testdata/tau-metrics-cached-validated/tau-metrics.bp 
  ProvDBPath = testdata/ProvDB.pkl 
  QueueSize = 40000 
  InterestFuncNum = 10 

  [Analyzer] 
  InputMatFile = testdata/testdata.mat 
  Dimension = 1 
  NumK = 10 
  KPar = 4 
  Bucket = 500 
  Width = 0  
  """
  possibleNumK=np.arange(5,30)
  possible_Kpar=np.arange(3,15)
  possible_bucket=np.arange(200,1000)

  text_file = open("chimbuko1.cfg", "w")
  n = text_file.write(parameters)
  text_file.close()

  scores,X_unique = MILOF("chimbuko.cfg")
  print("*",len(X))
  real_scores=[0 for i in X]
  scores =[1/i for i in scores]
  maxi=max(scores)
  for idx, x in enumerate(X):
    indices = [i for i, a in enumerate(X_unique.tolist()) if x == a]
    real_scores[idx]=scores[indices[0]]/maxi
  % cd ....
  return real_scores 



class class_MILOF:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
        
    
    
    
    def test(X,right,nbr_anomalies,gap):

        @jit
        def MILOF(X,Numk=10,KPar=4,Bucket=500):
            """
            LOF ne tient pas en compte les concentrations de points identiques sinon on tombe à un problème d'infini
            """
            import math
            sys.path.insert(0, '/home/amngobibin/Bureau/internship/stage/MILOF/lib')
            from MiLOF import MILOF
            scipy.io.savemat('MILOF/lib/test/testdata/testdata.mat', mdict={'DataStream': np.array(X)})
            parameters= """[Parser] 
            InputBPFile = testdata/tau-metrics-cached-validated/tau-metrics.bp 
            ProvDBPath = testdata/ProvDB.pkl 
            QueueSize = 40000 
            InterestFuncNum = 10 

            [Analyzer] 
            InputMatFile = testdata/testdata.mat 
            Dimension = 1 
            Width = 0
            NumK = 10 
            KPar = 4 
            Bucket = 500 
            """
            param="Numk="+str(10)+"\nKPar="+str(1)+"\nBucket="+str(1)
            parameters+=param
            text_file = open("MILOF/lib/test/chimbuko.cfg", "w")
            n = text_file.write(parameters)
            text_file.close()

            scores,X_unique = MILOF("MILOF/lib/test/chimbuko.cfg")
            print("*",len(X))
            real_scores=[0 for i in X]
            scores =[1/i for i in scores]
            maxi=max(scores)
            for idx, x in enumerate(X):
                indices = [i for i, a in enumerate(X_unique.tolist()) if x == a]
                real_scores[idx]=scores[indices[0]]/maxi
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
            return score
        
        def objective(args):
            print(args)
            scores= iforestASD(X,window_size=args["window_size"],n_estimators=args["n_estimators"]
                                )
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possibleNumK=np.arange(5,30)
        possible_Kpar=np.arange(3,15)
        possible_bucket=np.arange(200,1000)
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)
        , "n_estimators":hp.choice("n_estimators_index",possible_nbr_tree)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        end =time.monotonic()
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"window_size":possible_window_size[best["window_size_index"]],"n_estimator": possible_nbr_tree[best["n_estimators_index"]]}
        print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), str(best_param), end-start




# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 

#@jit                   
def test () :                                                         
    base_file ='Pattern_lengths_and_Number_of_discords2.xlsx'
    base = pd.read_excel(base_file)
    methods= {"iforestASD":class_iforestASD}# "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    
        
    for key, method in methods.items():
        thresholds=[]
        merlin_score=np.zeros(len(base))
        best_params = np.zeros(len(base))
        time_taken = np.zeros(len(base))
        best_params= ["params" for i in time_taken]
        for idx, dataset in enumerate(base["Dataset"]):
            df = pd.read_csv("dataset/"+dataset, names=["value"])
            print(dataset)
            if os.path.exists("real_nab_data/"+dataset) :
                start =time.monotonic()
                df = pd.read_csv("real_nab_data/"+dataset)
            column="value"
            if key=="test":
                plot_fig(df[column], title=dataset)
                continue##      
            # reading the dataset
            X =[[i] for i in df[column].values]

            right=np.array(str(base["Position discord"][idx]).split(';'))
            gap =int(base["discord length"][idx]) + int(int(base["Dataset length"][idx])/100)
            nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))


            if key =="iforestASD":
                real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_iforestASD.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key =="iforestASD_SUB":
                scores = iforestASD_SUB(X,500)
            if  key=="MILOF":# and "art_load_balancer_spikes.csv" in dataset:
                if len(np.unique(np.array(X),axis=0))<500:
                    print(len(X))
                    scores =[0.0 for i in X]
                else:
                    scores = method(X)
            if key=="HStree":
                scores = method(X)
            #calling methods
 
 
 
            
            df["anomaly_score"]=real_scores
            df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
            identified =identified
            print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )
            merlin_score[idx] = score# overlapping_merlin(identified,str(base["Position discord"][idx]).split(';'), int(int(base["Dataset length"][idx])/100) ) 
            directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_file_name=dataset.split('/')[-1]
            data_file_name =key+'_'+data_file_name
            dataset =directory+'/'+data_file_name
            df.to_csv(dataset, index=False)
            
            #thresholds.append(threshold)
            print("terminé")
            
            
            base[key+"_Overlap_merlin"] = merlin_score
            base[key+"best_param"] = best_params
            base[key+"time_taken"] = time_taken
            #base[key+"_threshold"]=thresholds
            base.to_excel("point_methods_result.xlsx")
            


test()


def overlapping_merlin(identified, expected,gap):
  score=0
  for expect in expected: 
    if identified in [*range (int(expect)-gap,int(expect)+gap)]:
      score+=1

  return score/len(expected)
