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

import math
import sys
from datetime import datetime
sys.path.append('MILOF/lib')
from MiLOF import MILOF

from test_Milof import class_MILOF
from drag_stream import class_our
from test_LAMP import class_LAMP
from test_hs_tree import class_hstree
from test_iforestASD import class_iforestASD
from score_nab import evaluating_change_point
from test_ARIMAFD import class_ARIMAFD
from test_KitNet import class_KitNet

"""This module permits to test methods on univariate datasets (Real known cause datasets of NAB)
Two scores are implemented in each method for the evaluation, the NAB score and the f1-score (over 1% the real position of the anomaly)

the best hyperparameters, the time taken by each method and the score with the best hyperparameters are recorded in the result file

"""


# Test pipeline   
#*****************************************************************************************************************************
import multiprocessing
mutex =multiprocessing.Lock()


from sklearn.preprocessing import OneHotEncoder

merlin_score=np.zeros(len(base))
time_taken = np.zeros(len(base))
best_params= ["params" for i in time_taken]
all_identified= ["no" for i in time_taken]

def dataset_test(merlin_score,best_params,time_taken,all_identified,key,idx,dataset,scoring_metric="merlin"):
    """This function insure test on each dataset, and it update the file containing the results

    Parameters passed contain each column of the result file for the method we are testing

    :param merlin_score: the merlin score of a method on each dataset
    :type merlin_score: list of float
    :param best_params: list of best parameters of the method on each dataset
    :type best_params: List of string
    :param time_taken: list of time taken by the method
    :type time_taken: List of float
    :param all_identified: List of list of anomaly identified per dataset
    :type all_identified: List
    :param key: the name of the method
    :type key: String
    :param idx: the id of the dataset we are currently testing  (dataset have idx in the list of datasets)
    :type idx: int
    :param dataset: the relative path of the dataset
    :type dataset: String
    :param scoring_metric: The scoring metric either NAB or MERLIN, defaults to "merlin"
    :type scoring_metric: str, optional
    :return: idx, best_param,time_taken_1, score, identified
    :rtype: int, String, float, float, List
    """
    
    if  True:# "ambient_temperature_system_failure" in dataset: #ligne =="params" or flag: 

        if sys.argv[2]=="U":
            base_file ='real_known_point_datasets.xlsx'
            base = pd.read_excel(base_file)
            
            df = pd.read_csv("dataset/"+dataset, names=["value"])
            print(dataset)
            if os.path.exists("real_nab_data/"+dataset) :
                df = pd.read_csv("real_nab_data/"+dataset)
            column="value"
            # reading the dataset
            X =[[i] for i in df[column].values]
            right=np.array(str(base["Position discord"][idx]).split(';'))
            nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))
        
        if sys.argv[2]=="M":
            base_file ='multivariate_abnormal_point.csv'
            base = pd.read_csv(base_file)
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
            # discord length/100
            gap =int(len(X)/100)
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

        
        df["anomaly_score"]=real_scores
        df["label"]=scores_label
        directory = os.path.dirname(('streaming_results/'+key+'/'+dataset))
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file_name=dataset.split('/')[-1]
        data_file_name =key+'_'+data_file_name
        dataset =directory+'/'+data_file_name
        df.to_csv(dataset, index=False)

        def insertion(file):
            best_params[idx]=best_param
            time_taken[idx]=time_taken_1
            merlin_score[idx] = score
            all_identified[idx] =identified
            try:
                base2 = pd.read_excel("f1score_"+scoring_metric+"_abnormal_point_results.xlsx") 
                base2[key+"_identified"] [idx]= all_identified[idx]
                base2[key+"_Overlap_merlin"] [idx]= score
                base2[key+"best_param"] [idx]=str(best_params [idx])
                base2[key+"time_taken"] [idx]= time_taken[idx]
                print("**********************************************************")
                print("***************",key,"********************")
                print("***************",dataset,"********************")
                print("**********************************************************")
                print( score, best_param, time_taken_1)
                print("**********************************************************")
            except :
                base2 = pd.read_excel("real_known_point_datasets.xlsx")
                base2[key+"_identified"] = all_identified
                base2[key+"_Overlap_merlin"] = merlin_score
                base2[key+"best_param"] =best_params 
                base2[key+"time_taken"]= time_taken
                
                if key in file:
                    print(best_params[idx], best_param)
                    for key2,value in best_params[idx].items():
                        base2["best_param"+key2] ="RAS"

            if key in file:
                base2.to_excel(file,index=False)
            else:
                base2.to_excel(file,index=False)
        insertion("result/f1score_"+scoring_metric+"_abnormal_point_results.xlsx")
        insertion("result/f1score_"+scoring_metric+"_"+key+"_abnormal_point_univariate.xlsx")
        return idx, best_param,time_taken_1, score, identified
    

import multiprocessing as mp
from multiprocessing import Manager
pool =mp.Pool(mp.cpu_count()-1)

def test (meth, type_dataset) :
    """This function test each method on all datasets

    The test on datasets are done in parallel through the multiprocessing library

    :param meth: the name of the method we will test
    :type meth: String
    """
                                                            
    methods= {meth:0}
    scoring_metric=["nab"] # ,"merlin"
    for key, method in methods.items():
        thresholds=[]
        
        for scoring  in scoring_metric:
            for i, d in enumerate(base["Dataset"]):
                dataset_test(merlin_score,best_params,time_taken,all_identified,key,i,base["Dataset"][i],scoring_metric=scoring)
            
            """with Manager() as mgr:
                merlin_score=mgr.list([]) + list(np.zeros(len(base)))
                time_taken = mgr.list([]) + list(np.zeros(len(base)))
                best_params= mgr.list([]) +  ["params" for i in time_taken]
                all_identified= mgr.list([]) + ["no" for i in time_taken]
                output =pool.starmap(dataset_test, [(merlin_score,best_params,time_taken,all_identified,
                key,idx,dataset,scoring) for idx,dataset in enumerate(base["Dataset"])  ] )
                print ("**** merlin score",merlin_score)
            """
import sys

print("***",sys.argv)

test(sys.argv[1],sys.argv[2])

