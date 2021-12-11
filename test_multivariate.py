# Import modules.
import numpy as np
from pysad.models import iforest_asd
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
import time
import os
import numba  
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
import math
import sys
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

sys.path.append('MILOF/lib')
from MiLOF import MILOF

from stream_discord import class_our
from test_LAMP import class_LAMP
from test_hs_tree import class_hstree
from test_iforestASD import class_iforestASD
from score_nab import evaluating_change_point
from test_ARIMAFD import class_ARIMAFD
from test_KitNet import class_KitNet

from test_Milof import class_MILOF

def dataset_test(key,idx,dataset,scoring_metric="merlin"):
    """This function assess a dataset among the set of dataset in the base file (real_known_cause_dataset),

    For further analysis, we put the label of each instance of each dataset in a csv file at streaming_result/name_of_the_method/dataset_name

    :param key: name of the method
    :type key: String
    :param idx: id of the line of the dataset
    :type idx: int
    :param dataset: name of the dataset
    :type dataset: String
    :param scoring_metric: scoring metric, defaults to "merlin"
    :type scoring_metric: str, optional
    """
    
    df = pd.read_csv("dataset/"+dataset)
    if True: # ligne =="params" or flag:

        
        print("we execute on ",dataset)
        oe_style = OneHotEncoder()
        for col in df.columns:
            if df.dtypes[col]==np.object:
                oe_results = oe_style.fit_transform(df[[col]])
                df=df.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))
        
        # reading the dataset
        X =[df.iloc[i].values for i in range(0,len(df))] 
        right=np.array(str(base["Position discord"][idx]).split(';'))
        nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))

        if scoring_metric=="merlin":
            gap =int(len(X)/100)
        if scoring_metric=="nab":
            gap = int(len(X)/(20*nbr_anomalies))
        if key =="HS-tree":
            hstree = class_hstree()
            real_scores, scores_label, identified,score,best_param, time_taken_1= hstree.test(dataset,X,right,nbr_anomalies,gap,scoring_metric=scoring_metric)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key =="MILOF":
            milof = class_MILOF()
            real_scores, scores_label, identified,score,best_param, time_taken_1= milof.test(dataset,X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="iforestASD":
            iforestASD= class_iforestASD()
            real_scores, scores_label, identified,score,best_param, time_taken_1= iforestASD.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="ARIMAFD":
            real_scores, scores_label, identified,score,best_param, time_taken_1= class_ARIMAFD.test(df,X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        if key=="KitNet":
            kitNet = class_KitNet()
            real_scores, scores_label, identified,score,best_param, time_taken_1= kitNet.test(X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
        df["anomaly_score"]=real_scores
        df["label"]=scores_label#[0 if i<threshold else 1 for i in scores ]
        
        
        directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file_name=dataset.split('/')[-1]
        data_file_name =key+'_'+data_file_name
        dataset =directory+'/'+data_file_name
        df.to_csv(dataset, index=False)

        file1=scoring_metric+"_abnormal_multivariate_point_results.xlsx"
        file2= scoring_metric+"_"+key+"_abnormal_multivarie_point.xlsx"

        return (key,file1,file2,idx, best_param,time_taken_1, score, identified) # key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified
    
import multiprocessing
mutex =multiprocessing.Lock()

base_file ='multivariate_abnormal_point.csv'
base = pd.read_csv(base_file)
import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(1)

thresholds=[]


        
# Test pipeline   
# ****************************************************************************************************************************
def test (meth) :                                                         
    merlin_score=np.zeros(len(base))
    time_taken = np.zeros(len(base))
    best_params= ["params" for i in time_taken]
    all_identified= ["no" for i in time_taken]
    methods= { meth:0}
    scoring_metric=["nab"] # you can also use NAB if you want to use the NAB score 
    for  key, method in methods.items() :
        
        with Manager() as mgr:
            def listener(m):
                print("*****************************************")
                print("*****************************************")
                key,file1,file2,idx, best_param,time_take, merlin_scor, identified=m
                all_identified[idx] =identified
                merlin_score[idx]=merlin_scor
                time_taken[idx]=time_take
                best_params[idx]=best_param
                file1="result/f1score_"+scoring+"_abnormal_multivariate_point_results.xlsx"
                file2= "f1score_"+scoring+"_"+key+"_abnormal_multivarie_point.xlsx"
                all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified)

            merlin_score=mgr.list(list(np.zeros(len(base)) ))
            time_taken =mgr.list(list(np.zeros(len(base)) ))
            best_params= mgr.list( ["params" for i in time_taken])
            all_identified= mgr.list( ["no" for i in time_taken])

            for scoring  in scoring_metric:
                
                for idx,dataset in enumerate(base["Dataset"]) :
                    """m=dataset_test(key,idx,dataset,scoring)
                    listener(m)"""
                    pool.apply_async(dataset_test, args=(key,idx,dataset,scoring,), callback=listener )
                pool.close()
                pool.join()
                file1=scoring+"_abnormal_multivariate_point_results.xlsx"
                file2= scoring+"_"+key+"_abnormal_multivarie_point.xlsx"


def all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified):
    insertion(file1,key,idx,best_params,time_taken,merlin_score, all_identified)
    insertion(file2,key,idx,best_params,time_taken,merlin_score, all_identified)

def insertion(file,key,idx,best_params,time_taken,merlin_score, all_identified):
            
            try:
                if key in file: 
                    base2 = pd.read_csv("streaming_results/"+file)
                else:
                    base2 = pd.read_csv(file) 
                
                base2[key+"_identified"] [idx]= all_identified[idx]
                base2[key+"_Overlap_merlin"] [idx]= merlin_score[idx]
                base2[key+"best_param"] [idx]=str(best_params [idx])
                base2[key+"time_taken"] [idx]= time_taken[idx]
            except :
                #if key in file: 
                base2 = pd.read_csv(base_file)
                """else:
                    base2 = pd.read_excel("merlin_abnormal_multivariate_point_results.xlsx") """
                base2[key+"_identified"] = all_identified
                base2[key+"_Overlap_merlin"] = merlin_score
                base2[key+"best_param"] =best_params 
                base2[key+"time_taken"]= time_taken
                if key in file:
                    for key2,value in best_params[idx].items():
                        base2["best_param"+key2] =best_params[idx][key2]

            if key in file:
                for key2,value in best_params[idx].items():
                    base2["best_param"+key2][idx] =best_params[idx][key2]
                base2.to_csv("streaming_results/"+file, index=False)
            else:
                base2.to_csv(file, index=False)


            
import sys
print("***",sys.argv)

test(sys.argv[1])





