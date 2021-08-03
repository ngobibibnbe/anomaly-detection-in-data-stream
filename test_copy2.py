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
        df = pd.read_csv("dataset/"+dataset)
        try: 
            base2 = pd.read_excel(scoring_metric+"_abnormal_multivariate_point_results.xlsx") 
            ligne = base2[key+"best_param"][idx]
        except :
            flag=True
            print("erreur de fichier ")
            ligne="erreur"

        #try :
        if True :#ligne =="params" or flag:
            oe_style = OneHotEncoder()
            for col in df.columns:
                if df.dtypes[col]==np.object:
                    oe_results = oe_style.fit_transform(df[[col]])
                    df=df.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))

            # reading the dataset
            X =[df.iloc[i].values for i in range(0,len(df))] #[[i] for i in df[column].values]
            right=np.array(str(base["Position discord"][idx]).split(';'))
            nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))
            #print(dataset)
            #print(right,"*",nbr_anomalies,"*",df.iloc[0])

            if scoring_metric=="merlin":
                gap =int(len(X)/100)
            if key =="HS-tree":
                real_scores, scores_label, identified,score,best_param, time_taken_1= class_hstree.test(dataset,X,right,nbr_anomalies,gap,scoring_metric=scoring_metric)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key =="MILOF":
                real_scores, scores_label, identified,score,best_param, time_taken_1= class_MILOF.test(dataset,X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key=="iforestASD":
                real_scores, scores_label, identified,score,best_param, time_taken_1= class_iforestASD.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key=="ARIMAFD":
                real_scores, scores_label, identified,score,best_param, time_taken_1= class_ARIMAFD.test(df,X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key=="KitNet":
                real_scores, scores_label, identified,score,best_param, time_taken_1= class_KitNet.test(X,right,nbr_anomalies,gap,scoring_metric="merlin")  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key=="LAMP":
                """base2 = pd.read_excel("point_methods_result_milof.xlsx")
                if base2[key+"best_param"][idx]=='params':
                    return idx, 0,0, 0, 0"""
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

            """best_params[key][idx]=best_param
            time_taken[key][idx]=time_taken_1
            merlin_score[key][idx] = score
            all_identified[key][idx] =identified"""
            #thresholds.append(threshold)

            """with mutex:
                #with open('point_methods_result_milof.xlsx') as csv_file:
                insertion(scoring_metric+"_abnormal_multivariate_point_results.xlsx")
                insertion(scoring_metric+"_"+key+"_abnormal_multivarie_point.xlsx")
                #csv_file.flush()"""
            # all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified)
            file1=scoring_metric+"_abnormal_multivariate_point_results.xlsx"
            file2= scoring_metric+"_"+key+"_abnormal_multivarie_point.xlsx"
            print(key,file1,file2,idx, best_param,time_taken_1, score, identified)

            return (key,file1,file2,idx, best_param,time_taken_1, score, identified) # key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified
        """except:
            file1=scoring_metric+"_abnormal_multivariate_point_results.xlsx"
            file2= scoring_metric+"_"+key+"_abnormal_multivarie_point.xlsx"
            #q.put((key,file1,file2,idx, best_params,time_taken, merlin_sco
            return (key,file1,file2,idx, {"Erreur":"RAS"},"RAS", "RAS", "RAS")"""

import multiprocessing
mutex =multiprocessing.Lock()

base_file ='multivariate_abnormal_point.csv'
base = pd.read_csv(base_file)
import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(mp.cpu_count())

thresholds=[]



# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 
#*****************************************************************************************************************************


def test () :                                                         
    """    merlin_score=np.zeros(len(base))
        time_taken = np.zeros(len(base))
        best_params= ["params" for i in time_taken]
        all_identified= ["no" for i in time_taken]
        
        """
    methods= { "HS-tree":0, "MILOF":0}#"ARIMAFD":0}#, "HS-tree":0, "iforestASD":0}#"MILOF":0}# "MILOF":class_MILOF.test, "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    scoring_metric=["merlin"] # ,"merlin"
    for  key, method in methods.items() :

        with Manager() as mgr:
            def listener(m):
                print("*****************************************")
                print(m)
                print("*****************************************")
                key,file1,file2,idx, best_param,time_take, merlin_scor, identified=m
                all_identified[idx] =identified
                merlin_score[idx]=merlin_scor
                time_taken[idx]=time_take
                best_params[idx]=best_param
                print(best_params,"***")
                file1=scoring+"_abnormal_multivariate_point_results.xlsx"
                file2= scoring+"_"+key+"_abnormal_multivarie_point.xlsx"
                all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified)
            """merlin_score=mgr.dict()
            time_taken =mgr.dict()
            best_params= mgr.dict()
            all_identified=mgr.dict()
            for key , method in methods.items():
                merlin_score[key]=mgr.list(list(np.zeros(len(base)) ))
                time_taken[key] =mgr.list(list(np.zeros(len(base)) ))
                best_params [key]= mgr.list( ["params" for i in time_taken])
                all_identified[key] = mgr.list( ["no" for i in time_taken])
                """
            merlin_score=mgr.list(list(np.zeros(len(base)) ))
            time_taken =mgr.list(list(np.zeros(len(base)) ))
            best_params= mgr.list( ["params" for i in time_taken])
            all_identified= mgr.list( ["no" for i in time_taken])

            for scoring  in scoring_metric:
                """for i, d in enumerate(base["Dataset"]):
                    dataset_test(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring_metric=scoring)
                """
                for idx,dataset in enumerate(base["Dataset"]) :
                    pool.apply_async(dataset_test, args=(key,idx,dataset,scoring,), callback=listener )
                pool.close()
                pool.join()
                file1=scoring+"_abnormal_multivariate_point_results.xlsx"
                file2= scoring+"_"+key+"_abnormal_multivarie_point.xlsx"
                all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified)
                #output =pool.apply_async(dataset_test, [(merlin_score,best_params,time_taken,all_identified,key,idx,dataset,scoring) for idx,dataset in enumerate(base["Dataset"])  ], callback=listener )

                """manager =multiprocessing.Manager()
                #q=manager.Queue()
                pool=multiprocessing.Pool()
                for idx,dataset in enumerate(base["Dataset"]) :
                    pool.apply_async(dataset_test, args=(key,idx,dataset,scoring,), callback=listener )
                pool.close()
                pool.join()
                file1=scoring+"_abnormal_multivariate_point_results.xlsx"
                file2= scoring+"_"+key+"_abnormal_multivarie_point.xlsx"
                all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified)"""


def all_insertion(key,file1,file2,idx, best_params,time_taken, merlin_score, all_identified):
    print(file1,key,idx,best_params,time_taken,merlin_score, all_identified)
    insertion(file1,key,idx,best_params,time_taken,merlin_score, all_identified)
    insertion(file2,key,idx,best_params,time_taken,merlin_score, all_identified)

def insertion(file,key,idx,best_params,time_taken,merlin_score, all_identified):

            try:
                if key in file: 
                    base2 = pd.read_excel("streaming_results/"+file)
                else:
                    base2 = pd.read_excel(file) 

                base2[key+"_identified"] [idx]= all_identified[idx]
                base2[key+"_Overlap_merlin"] [idx]= merlin_score[idx]
                base2[key+"best_param"] [idx]=str(best_params [idx])
                base2[key+"time_taken"] [idx]= time_taken[idx]
            except :
                if key in file: 
                    base2 = pd.read_csv(base_file)
                else:
                    base2 = pd.read_excel("merlin_abnormal_multivariate_point_results.xlsx") 
                base2[key+"_identified"] = all_identified
                base2[key+"_Overlap_merlin"] = merlin_score
                base2[key+"best_param"] =best_params 
                base2[key+"time_taken"]= time_taken
                print("***************************************except***************")
                if key in file:
                    print(best_params)
                    for key2,value in best_params[idx].items():
                        base2["best_param"+key2] =best_params[idx][key2]

            if key in file:
                for key2,value in best_params[idx].items():
                    base2["best_param"+key2][idx] =best_params[idx][key2]
                base2.to_excel("streaming_results/"+file, index=False)
            else:
                base2.to_excel(file, index=False)




test()

