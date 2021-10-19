import scipy.io
from river import drift
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import os 
from hyperopt import fmin, tpe,hp, STATUS_OK, Trials
import numpy as np
import matrixprofile as mp
import time
from score_nab import evaluating_change_point
import matrixprofile as mp
#from matrixprofile import *
from saxpy.hotsax import find_discords_hotsax
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


class class_LAMP:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    
    def test(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        
      def LAMP(X,training_size,w=gap):
            #diviser le dataframe 
            ts_train =X[:training_size]
            ts_val =X[: training_size]#int(3*len(X)/8)]
            ts_test = X#X[int(3*len(X)/8):]
            mp_train = mp.compute(ts_train,w)['mp']
            mp_test = mp.compute(ts_test,w)['mp']
            mp_val = mp.compute(ts_val,w)['mp']
            mat={}
            mat["ts_train"]=ts_train.reshape(-1,1)
            #print(mat["ts_train"])
            mat["ts_test"] =ts_test.reshape(-1,1)
            mat["ts_val"]=ts_val.reshape(-1,1)
            mat["mp_test"]=mp_test.reshape(-1,1)
            mat["mp_train"]=mp_train.reshape(-1,1)
            mat["mp_val"]=mp_val.reshape(-1,1)
            try :
                scipy.io.savemat("dataset/test_tmp.mat", mat)
                os.system("python3 LAMP-conference_code/train_neural_net_LAMP.py "+str(w)+" dataset/test_tmp.mat ./logs "+dataset) # remplacer 100 par la vrai taille de fenêtre
                scores =pd.read_csv('LAMP-conference_code/predict/predicted_matrix_profile_'+dataset+'.txt', sep=" ", header=None, names=["column"] )
                #os.system("rm -r logs")
                #print(scores, "**")
                scores=list(scores["column"].values)
                print("**************************************",len(scores), len(ts_test), len(mp_test), len(ts_val) )
                scores =scores+list(np.zeros(len(ts_test)-len(scores) ))
            except:
                scores =list(np.zeros(len(ts_test)))
            print(len(scores), len(ts_train), len(ts_test), len(mp_test), len(ts_val) )
            
            return scores

                        
        
      def scoring(scores):
            identified =np.where(scores.squeeze()==1)[0]
            sub_identified=np.array([])
            sub_right=np.array([])
            for identify in identified:
                sub_identified=np.concatenate([sub_identified,np.arange(identify, identify+gap)])
            for identify in right:
                identify =int(identify)
                sub_right=np.concatenate([sub_right,np.arange(identify, identify+gap)])
            sub_identified =np.unique(sub_identified)
            sub_right =np.unique(sub_right)  
            recall =len(np.intersect1d(sub_identified,sub_right))/len(sub_right)
            precision = len(np.intersect1d(sub_identified,sub_right))/len(sub_identified)
            try :
                score =2*(recall*precision)/(recall+precision) 
            except :
                score=0.0  
            return score 


        
      def score_to_label(nbr_anomalies,scores,gap):
        
        thresholds = np.unique(scores)
        max =np.amax(thresholds)
        thresholds = sorted(thresholds, reverse=True)
        thresholds =thresholds[:min(len(thresholds),30)]

        f1_scores =[]
        
        for threshold in thresholds:
            labels=np.where(scores<threshold,0,1)
            a=scoring(labels)
            f1_scores.append(a) 
        q = list(zip(f1_scores, thresholds))
        thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
        threshold=thres
        arg=np.where(thresholds==thres)
        return np.where(scores<threshold,0,1)# i will throw only real_indices here. [0 if i<threshold else 1 for i in scores ]



      right_discord =[ int(discord) for discord in right]
      start =time.monotonic()
      real_scores= LAMP(X,training_size=int(len(X)/4) )
      end =time.monotonic()       
      best_param={"window":int(len(X)/4) }
      if real_scores == [1/(1+i) for i in list(np.zeros(len(X)))]:
          best_param={"window":"error" }
      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start 



    
    def test_mp(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        
      def Matrix_profile(dataset,nbr_of_discord,n=gap):
            df = pd.read_csv("dataset/"+dataset, sep="  ", header=None, names=["column1"])
            column="column1"
            profile = mp.compute(df[column].values,n)
            discords = mp.discover.discords(profile, exclusion_zone=int(n/2), k=nbr_of_discord)["discords"]
            scores =np.zeros(len(X))
            scores[discords]=1
            return scores

                        
        
      def scoring(scores):
            identified =np.where(scores.squeeze()==1)[0]
            sub_identified=np.array([])
            sub_right=np.array([])
            for identify in identified:
                sub_identified=np.concatenate([sub_identified,np.arange(identify, identify+gap)])
            for identify in right:
                identify =int(identify)
                sub_right=np.concatenate([sub_right,np.arange(identify, identify+gap)])
            sub_identified =np.unique(sub_identified)
            sub_right =np.unique(sub_right)  
            recall =len(np.intersect1d(sub_identified,sub_right))/len(sub_right)
            precision = len(np.intersect1d(sub_identified,sub_right))/len(sub_identified)
            try :
                    score =2*(recall*precision)/(recall+precision) 
            except :
                score=0.0  
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
          #try:
          scores= Matrix_profile(dataset,nbr_of_discord= args["nbr_anomalies"], n=gap)
          scores =score_to_label(nbr_anomalies,scores,gap)
          return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}

      possible_nbr_anomalies=np.arange(max(1,nbr_anomalies-1),nbr_anomalies+3)
      space2 ={"nbr_anomalies":hp.choice("nbr_anomalies_index",possible_nbr_anomalies)}
      trials = Trials()
      
      
      #best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)


      best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=1,trials = trials)
      
      start =time.monotonic()
      real_scores= Matrix_profile(dataset,nbr_of_discord=possible_nbr_anomalies[best["nbr_anomalies_index"]],n=gap )
      end =time.monotonic()
      
          
      best_param={"nbr_of_discord":possible_nbr_anomalies[best["nbr_anomalies_index"]] }

      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start 



    def test_hotsax(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):


      def hotsax(dataset,nbr_of_discord,n,w,a):
            df = pd.read_csv("dataset/"+dataset, sep="  ", header=None, names=["column1"])
            column="column1"
            dd =df[column].to_numpy() #genfromtxt("data/ecg0606_1.csv", delimiter=',') 
            print("***",dd,"len discord:",n, "m_discords:",nbr_of_discord,a,w )
            discords = find_discords_hotsax(dd, win_size=n, num_discords=nbr_of_discord, a_size=a,paa_size=w)
            d=[]
            for discord in discords:
                d.append(discord[0])
            scores =np.zeros(len(X))
            scores[d]=1
            print("*****************check scores",scores)
            return scores               
        
      def scoring(scores):
            identified =np.where(scores.squeeze()==1)[0]
            sub_identified=np.array([])
            sub_right=np.array([])
            for identify in identified:
                sub_identified=np.concatenate([sub_identified,np.arange(identify, identify+gap)])
            for identify in right:
                identify =int(identify)
                sub_right=np.concatenate([sub_right,np.arange(identify, identify+gap)])
            sub_identified =np.unique(sub_identified)
            sub_right =np.unique(sub_right)  
            recall =len(np.intersect1d(sub_identified,sub_right))/len(sub_right)
            precision = len(np.intersect1d(sub_identified,sub_right))/len(sub_identified)
            try :
                    score =2*(recall*precision)/(recall+precision) 
            except :
                score=0.0  
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
          #try:
          scores= hotsax(dataset,nbr_of_discord=args["nbr_anomalies"],n=gap,w=args["paa_size"],a=args["nbr_symbols"])
          scores =score_to_label(nbr_anomalies,scores,gap)
          return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}

      possible_nbr_anomalies=np.arange(max(1,nbr_anomalies-1),nbr_anomalies+3)
      possible_paa=np.arange(3,15)
      possible_symbols=np.arange(3,15)
      space2 ={"nbr_anomalies":hp.choice("nbr_anomalies_index",possible_nbr_anomalies),
      "paa_size":hp.choice("paa_size_index",possible_paa), "nbr_symbols":hp.choice("nbr_symbols_index",possible_paa)}
      trials = Trials()
      
      
      best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
      #print(best)
      start =time.monotonic()
      real_scores= hotsax(dataset,nbr_of_discord=best["nbr_anomalies_index"],n=gap,w=best["paa_size_index"],a=best["nbr_symbols_index"] )
      end =time.monotonic()
      
          
      best_param={"nbr_of_discord":possible_nbr_anomalies[best["nbr_anomalies_index"]],"paa_size":best["paa_size_index"],"number_symbols":best["nbr_symbols_index"]  }
      """if real_scores == [1/(1+i) for i in list(np.zeros(len(X)))]:
          best_param={"The model is porviding null every where" }"""
      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start
