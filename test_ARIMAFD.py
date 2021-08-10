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

def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        real_indice=int(real_indice)
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice-gap,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag


class class_ARIMAFD:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
            
    def test(df,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

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
            #print(a.evaluate_nab([[0.1,1]]))
            #print(a.generate_tensor(ar_order=100)[:,0,0].shape)
            scores=np.concatenate([np.zeros(AR_window_size),a.generate_tensor(ar_order=AR_window_size)[:,0,0].squeeze()])# a.bin_metric]) # Joining arr1 and arr2
            scores=np.concatenate([scores,np.zeros(d)])# a.bin_metric]) # Joining arr1 and arr2
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
            precision =precision/len(identified) # la methode est precise si elle arrive à plus trouveer de zones anormales que non normales 
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
            scores= ARIMAFD(X,AR_window_size=args["AR_window_size"],d=args["d"])
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        #possible_nbr_tree =np.arange(1,25)#[*range(1,100)]
        possible_AR_window_size =np.arange(10, 200) #[*range(200,1000)]
        possible_d =np.arange(1,10)
        space2 ={"AR_window_size":hp.choice("AR_window_size_index",possible_AR_window_size), "d":hp.choice("d_index",possible_d) }
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= ARIMAFD(X,AR_window_size=possible_AR_window_size[best["AR_window_size_index"]],d=possible_d[best["d_index"]])
        end =time.monotonic()
        #print(real_scores)
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"AR_window_size":possible_AR_window_size[best["AR_window_size_index"]] , "differencing-order":possible_d[best["d_index"]]}
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
            #print(dataset)
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
            #print(len(identified),score)
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

