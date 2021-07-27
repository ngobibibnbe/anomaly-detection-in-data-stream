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


class class_ARIMAFD:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
            
    def test(df,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        def ARIMAFD(X,window_size):
            X =np.array(X)
            X=X#.reshape(-1,1)
            X=pd.DataFrame(X,columns=df.columns)
            """
            
            """
            a =anomaly_detection(X)
            a.tensor =a.generate_tensor(ar_order=window_size)
            a.proc_tensor(No_metric=1)
            #print(a.evaluate_nab([[0.1,1]]))
            #print(a.generate_tensor(ar_order=100)[:,0,0].shape)
            scores=np.concatenate([np.zeros(window_size),a.generate_tensor(ar_order=window_size)[:,0,0].squeeze()])# a.bin_metric]) # Joining arr1 and arr2
            print(len(X),"****",len(scores), "***", scores)
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
            scores= ARIMAFD(X,window_size=args["window_size"])
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        #possible_nbr_tree =np.arange(1,25)#[*range(1,100)]
        possible_window_size =np.arange(20, 200) #[*range(200,1000)]
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= ARIMAFD(X,window_size=possible_window_size[best["window_size_index"]])
        end =time.monotonic()
        print(real_scores)
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"window_size":possible_window_size[best["window_size_index"]] }
        #print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start






# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 

#@jit                   
def test () :                                                         
    base_file ='abnormal_point_datasets.xlsx'
    base = pd.read_excel(base_file)
    methods= {"ARIMA":0}# "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    
        
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
            gap = int(int(base["Dataset length"][idx])/100)
            nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))


            real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_ARIMAFD.test(X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            print(len(identified),score)
            break 
 
 
 
            
            df["anomaly_score"]=real_scores
            df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
            identified =identified
            #print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )
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
            


#test()

