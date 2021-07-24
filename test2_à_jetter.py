
# Import modules.
from sklearn.utils import shuffle
#from pysad.evaluation import AUROCMetric

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
actual_dataset=[0]


from test_1 import class_iforestASD
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

class class_MILOF:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(dataset,X,right,nbr_anomalies,gap):

        def MILOF_(X,NumK,KPar,Bucket ):
            scipy.io.savemat('MILOF/lib/test/testdata/testdata.mat', mdict={'DataStream': np.array(X)})
            parameters ="""[Parser]
            InputBPFile = testdata/tau-metrics-cached-validated/tau-metrics.bp
            ProvDBPath = testdata/ProvDB.pkl
            QueueSize = 40000
            InterestFuncNum = 10

            [Analyzer]
            InputMatFile = MILOF/lib/test/testdata/testdata.mat
            Dimension = 1
            Width = 0
            """
            param =  "\n"+"Bucket="+str(Bucket)
            parameters+="NumK="+str(NumK)
            parameters+="\n"+"Kpar="+str(KPar)
            parameters+=param
            #print(parameters)
            file_name =dataset[0]
            text_file = open("MILOF/conf/"+file_name+".cfg", "w")
            n = text_file.write(parameters)
            text_file.close()

            scores,X_unique = MILOF("MILOF/conf/"+file_name+".cfg")
            #print("*",len(X))
            real_scores=[0 for i in X]
            scores =[1/i for i in scores]
            maxi=max(scores)
            #print("**********************unique",len(np.unique(np.array(X))), "score len",len(scores), "real size", len(X))
            
            for idx, x in enumerate(X):
                indices = [i for i, a in enumerate(np.array(X_unique) ) if x == a]
                #print("**",indices)
                #print("**",indices[0],len(scores))
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
            return score
        
        def objective(args):
            print(args)
            
            #try:
            scores= MILOF_(X,NumK=args["NumK"],KPar=args["KPar"],Bucket=args["Bucket"] )
            """except :
                print("there was an error so")
                scores=np.zeros(len(X))"""
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


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
            best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=1,trials = trials)
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
        return real_scores, scores_label, identified,scoring(scores_label), str(best_param), end-start







base_file ='point_methods_result_milof.xlsx'
base = pd.read_excel(base_file)

def dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset):

    df = pd.read_csv("dataset/"+dataset, names=["value"])
    print(dataset)
    if os.path.exists("real_nab_data/"+dataset) :
        df = pd.read_csv("real_nab_data/"+dataset)
    column="value"
    if key=="test":
        plot_fig(df[column], title=dataset)
    # reading the dataset
    X =[[i] for i in df[column].values]
    right=np.array(str(base["Position discord"][idx]).split(';'))
    gap =int(base["discord length"][idx]) + int(int(base["Dataset length"][idx])/100)
    nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))

    if key =="MILOF":
        real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_MILOF.test(dataset,X,right,nbr_anomalies,gap)
    if key=="iforestASD":
        real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_iforestASD.test(X,right,nbr_anomalies,gap)
    df["anomaly_score"]=real_scores
    df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
    all_identified[idx] =identified
    print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )
    merlin_score[idx] = score
    directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_file_name=dataset.split('/')[-1]
    data_file_name =key+'_'+data_file_name
    dataset =directory+'/'+data_file_name
    df.to_csv(dataset, index=False)
    
    #thresholds.append(threshold)
    print("termine")
    
    base[key+"_identified"] = all_identified
    base[key+"_Overlap_merlin"] = merlin_score
    base[key+"best_param"] = best_params
    base[key+"time_taken"] = time_taken
    #base[key+"_threshold"]=thresholds
    base.to_excel("point_methods_result_milof.xlsx")
    
import multiprocessing as mp
pool =mp.Pool(mp.cpu_count())

def test () :                                                         
    
    methods= {"MILOF":class_MILOF.test}# "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
        
    for key, method in methods.items():
        thresholds=[]
        merlin_score=np.zeros(len(base))
        time_taken = np.zeros(len(base))
        best_params= ["params" for i in time_taken]
        all_identified= ["no" for i in time_taken]

        #pool.starmap(dataset_test, [(merlin_score,best_params,time_taken,all_identified,key,idx,dataset) for idx,dataset in enumerate(base["Dataset"])] )
        for idx, dataset in enumerate(base["Dataset"]):
            if base["MILOFbest_param"][idx]=="params":
                print("**********",base["MILOFbest_param"][idx])
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset)
            """df = pd.read_csv("dataset/"+dataset, names=["value"])
            
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
                real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_MILOF.test(X,right,nbr_anomalies,gap)            if key =="iforestASD_SUB":
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
            
            
            base[key+"_Overlap_merlin"] = merlin_score
            base[key+"best_param"] = best_params
            base[key+"time_taken"] = time_taken
            #base[key+"_threshold"]=thresholds
            base.to_excel("point_methods_result_milof.xlsx")"""
            


test()


