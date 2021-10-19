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
    """
    The idea of this check is to insure two neighbours anomalies are not throw since we are using the merlin score and the nab one. Let a method
    give all its anomalies in the same window is not realy fair. 
    """
    Flag=True
    for real_indice in real_indices:
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice-gap,real_indice+gap) # " il y'avait -gap
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
  return np.where(scores<threshold,0,1)# i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]


#ok



import math
import sys
from datetime import datetime
from score_nab import evaluating_change_point
class class_KitNet:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(dataset,X,right,nbr_anomalies,gap, scoring_metric="merlin"):

        #@jit
        def HStree(X, initial_window=500, window_size=500, num_trees=25, max_depth=15):
            """
            Malheureusement le concept drift n'est pas encore implémenté dans pysad nous devons le faire manuellement
            """
            np.random.seed(61)  # Fix random seed.
            X_all =np.array(X)
            iterator = ArrayStreamer(shuffle=False)
            A=X_all.reshape(-1,1)
            # Fit reference window integration to first 100 instances initially.
            model=models.KitNet(num_features =window_size, grace_feature_mapping=window_size,grace_anomaly_detector =window_size)
            scores=[]
            #scores= scores+ np.zeros(len(X_all[:initial_window])).tolist()
            for x in tqdm(iterator.iter(X_all)):
                model.fit_partial(x)  # Fit to the instance.
                score = model.score_partial(x)  # Score the instance.
                scores.append(score)
            print(len(scores))
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
                score=nab_score       
            return score
        
        def objective(args):
            print(args)
            
            #try:
            scores= HStree(X,initial_window=args["initial_window"],window_size=args["window_size"] )
            scores= score_to_label(nbr_anomalies,scores,gap) 
            
            
            return 1/(1+scoring(scores))#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_initial_window=np.arange(500,1000)#[*range(1,100)]
        possible_window_size =np.arange(300, 1000 ) #[*range(200,1000)]
        space2 ={"initial_window":hp.choice("initial_window_index",possible_initial_window)
        , "window_size":hp.choice("window_size_index",possible_window_size)}
        trials = Trials()
        
        
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=1,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= HStree(X,initial_window=possible_initial_window[best["initial_window_index"]],window_size=possible_window_size[best["window_size_index"]] )
        end =time.monotonic()
        best_param={"initial_window":possible_initial_window[best["initial_window_index"]],"window_size":possible_window_size[best["window_size_index"]] }

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


        


import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(mp.cpu_count())

def test () :                                                         
    
    methods= {"hstree":0}# "MILOF":class_MILOF.test, "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
        
    for key, method in methods.items():
        thresholds=[]
        merlin_score=np.zeros(len(base))
        time_taken = np.zeros(len(base))
        best_params= ["params" for i in time_taken]
        all_identified= ["no" for i in time_taken]

        with Manager() as mgr:
            merlin_score=mgr.list([]) + list(np.zeros(len(base)))
            time_taken = mgr.list([]) + list(np.zeros(len(base)))
            best_params= mgr.list([]) +  ["params" for i in time_taken]
            all_identified= mgr.list([]) + ["no" for i in time_taken]
            output =pool.starmap(dataset_test, [(merlin_score,best_params,time_taken,all_identified,key,idx,dataset) for idx,dataset in enumerate(base["Dataset"])  ] )
        

        print(output[0])
        """print(len(output))
        
        base[key+"_identified"] = all_identified
        base[key+"_Overlap_merlin"] = merlin_score
        base[key+"best_param"] =best_params 
        base[key+"time_taken"]= time_taken
        for idx, best_param,time_taken, score, identified in output:
            #base = pd.read_excel(base_file)
            base[key+"_identified"] [idx]= identified
            base[key+"_Overlap_merlin"] [idx]= score
            base[key+"best_param"] [idx]=best_param
            base[key+"time_taken"] [idx]= time_taken
            base.to_excel("point_methods_result_milof.xlsx")"""

        """for idx, dataset in enumerate(base["Dataset"]):
            if base["MILOFbest_param"][idx]=="params":
                #print("**********",base["MILOFbest_param"][idx])
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset)
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


            if key =="MILOF":
                real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_MILOF.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key =="iforestASD_SUB":
                scores = iforestASD_SUB(X,500)
            #""if  key=="MILOF":# and "art_load_balancer_spikes.csv" in dataset:
                if len(np.unique(np.array(X),axis=0))<500:
                    print(len(X))
                    scores =[0.0 for i in X]
                else:
                    scores = method(X)""
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
            base.to_excel("point_methods_result_milof.xlsx")"""
            


#test()


def overlapping_merlin(identified, expected,gap):
  score=0
  for expect in expected: 
    if identified in [*range (int(expect)-gap,int(expect)+gap)]:
      score+=1

  return score/len(expected)
