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
            Dimension = 1
            Width = 0
            """
            param =  "\n"+"Bucket="+str(Bucket)
            parameters+="NumK="+str(NumK)
            parameters+="\n"+"Kpar="+str(KPar)+"\nInputMatFile="+input_file
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
        possible_Bucket =np.array([min(500,len(np.unique(np.array(X)))) ] )
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


        




# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 
#*****************************************************************************************************************************
import multiprocessing
mutex =multiprocessing.Lock()

base_file ='abnormal_point_datasets.xlsx'
base = pd.read_excel(base_file)

merlin_score=np.zeros(len(base))
time_taken = np.zeros(len(base))
best_params= ["params" for i in time_taken]
all_identified= ["no" for i in time_taken]

def dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset,scoring_metric="merlin"):

    try: 
        base2 = pd.read_excel(scoring_metric+"_abnormal_multivariate_point_results.xlsx") 
        ligne = base2[key+"best_param"][idx]
    except :
        flag=True
        print("erreur de fichier ")
        ligne="erreur"
        
    #try :
    if ligne =="params" or flag:

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
        nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))

        if scoring_metric=="merlin":
            #gap =int(int(base["Dataset length"][idx])/100)
            # discord length
            gap =int(int(base["discord length"][idx])/2)
        if scoring_metric=="nab":
            gap = int(len(X)/(20*nbr_anomalies))
        
        if key =="HS-tree":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_hstree.test(dataset,X,right,nbr_anomalies,gap,scoring_metric=scoring_metric)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        if key =="MILOF":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_MILOF.test(dataset,X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="iforestASD":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_iforestASD.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="ARIMAFD":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_ARIMAFD.test(df[[column]],X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="KitNet":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_KitNet.test(X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        if key=="LAMP":
            base2 = pd.read_excel("point_methods_result_milof.xlsx")
            if base2[key+"best_param"][idx]=='params':
                return idx, 0,0, 0, 0
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_LAMP.test(dataset,df[column].values,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        if key=="our":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_our.test(dataset,df[column].values,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme

        df["anomaly_score"]=real_scores
        df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
        
        #print("anomalies at:" , identified, "while real are at:", str(base["Position discord"][idx]) )
        
        directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file_name=dataset.split('/')[-1]
        data_file_name =key+'_'+data_file_name
        dataset =directory+'/'+data_file_name
        df.to_csv(dataset, index=False)
        
        #thresholds.append(threshold)
        print("terminé")
        def insertion(file):
            best_params[idx]=best_param
            time_taken[idx]=time_taken_1
            merlin_score[idx] = score
            all_identified[idx] =identified
            try:
                
                base2 = pd.read_excel(file) 
                
                base2[key+"_identified"] [idx]= all_identified[idx]
                base2[key+"_Overlap_merlin"] [idx]= score
                base2[key+"best_param"] [idx]=str(best_params [idx])
                base2[key+"time_taken"] [idx]= time_taken[idx]
            except :
                base2 = pd.read_excel("abnormal_point_datasets.xlsx")
                base2[key+"_identified"] = all_identified
                base2[key+"_Overlap_merlin"] = merlin_score
                base2[key+"best_param"] =best_params 
                base2[key+"time_taken"]= time_taken
                
                if key in file:
                    print(best_params[idx], best_param)
                    for key2,value in best_params[idx].items():
                        base2["best_param"+key2] ="RAS"

            if key in file:
                for key2,value in best_params[idx].items():
                    base2["best_param"+key2][idx] =best_params[idx][key2]
                base2.to_excel(file)
            else:
                base2.to_excel(file)




        with mutex:
            with open('abnormal_point_datasets.xlsx') as csv_file:
            #***insertion(scoring_metric+"_abnormal_point_results.xlsx")
                insertion("result/"+scoring_metric+"_"+key+"_abnormal_point_univariate.xlsx")
                csv_file.flush()
    return idx, best_param,time_taken_1, score, identified
    

import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(mp.cpu_count())

def test (meth) :                                                         
    
    methods= {meth:0}#, "HS-tree":0,"MILOF":0,"HS-tree":0, "iforestASD":0}#"MILOF":0}# "MILOF":class_MILOF.test, "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    scoring_metric=["merlin"] # ,"merlin"
    for key, method in methods.items():
        thresholds=[]
        
        for scoring  in scoring_metric:
            #dataset_test(merlin_score,best_params,time_taken,all_identified,key,1,base["Dataset"][1],scoring_metric=scoring)

            for i, d in enumerate(base["Dataset"]):
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring_metric=scoring)
        
            """with Manager() as mgr:
                merlin_score=mgr.list([]) + list(np.zeros(len(base)))
                time_taken = mgr.list([]) + list(np.zeros(len(base)))
                best_params= mgr.list([]) +  ["params" for i in time_taken]
                all_identified= mgr.list([]) + ["no" for i in time_taken]
                output =pool.starmap(dataset_test, [(key,idx,dataset,scoring) for idx,dataset in enumerate(base["Dataset"])  ] )
                print ("**** merlin score",merlin_score)"""

