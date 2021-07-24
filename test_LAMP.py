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




class class_LAMP:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    def LAMP(self,X,w):
        #diviser le dataframe 
        ts_train =X[:int(len(X)/4)]
        ts_val =X[int(len(X)/4) : ]#int(3*len(X)/8)]
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
        scipy.io.savemat("dataset/test_tmp.mat", mat)
        os.system("python3 LAMP-conference_code/train_neural_net_LAMP.py "+str(w)+" dataset/test_tmp.mat ./logs") # remplacer 100 par la vrai taille de fenêtre
        scores =pd.read_csv('predicted_matrix_profile.txt', sep=" ", header=None, names=["column"] )
        #os.system("rm -r logs")
        #print(scores, "**")
        scores=list(scores["column"].values)
        scores =[1/(i+1) for i in scores]
        print(len(scores), len(ts_train), len(ts_test), len(mp_test), len(ts_val) )
        scores =scores+list(np.zeros(len(ts_test)-len(scores) ))
        print(len(scores), len(ts_train), len(ts_test), len(mp_test), len(ts_val) )
        return scores

    
    def test(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        
      def LAMP(X,w):
        #diviser le dataframe 
        ts_train =X[:int(len(X)/4)]
        ts_val =X[int(len(X)/4) : ]#int(3*len(X)/8)]
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
          scores= LAMP(X,w=args["window"])
          scores =score_to_label(nbr_anomalies,scores,gap)
          

          return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


      possible_window=np.arange(100,gap+200)
      space2 ={"window":hp.choice("window_index",possible_window)}
      trials = Trials()
      
      
      best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
      #print(best)
      start =time.monotonic()
      real_scores= LAMP(X,w=possible_window[best["window_index"]] )
      end =time.monotonic()
      
          
      best_param={"window":possible_window[best["window_index"]] }
      if real_scores == [1/(1+i) for i in list(np.zeros(len(X)))]:
          best_param={"window":"error" }

      """except :
          print("there was an error")
          best_param={"Numk":"RAS","KPar":"RAS","Bucket_index":"RAS" }
      """
        #real_scores=np.zeros(len(X))
      #iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
      
      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      #print("the final score is", scoring(scores_label),identified)
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start


        


"""
    
base_file ='Pattern_lengths_and_Number_of_discords2.xlsx'
base = pd.read_excel(base_file)
merlin_score=np.zeros(len(base))
best_params = np.zeros(len(base))
time_taken = np.zeros(len(base))
best_params= ["params" for i in time_taken]"""
"""for idx, dataset in enumerate(base["Dataset"]):
    df = pd.read_csv("dataset/"+dataset, names=["value"])
    if os.path.exists("real_nab_data/"+dataset) :
        df = pd.read_csv("real_nab_data/"+dataset)
    print(dataset)
    #if dataset=="nab-data/artificialWithAnomaly/art_daily_flatmiddle.csv":
    column="value"
    X =[[i] for i in df[column].values]
    column="value"
    #print(df[column].values) X,right,nbr_anomalies,gap
    scores = class_LAMP.test(dataset,df[column].values,[2000],1,200)
    print("****",len(scores), len(X))
#break    # reading the dataset
"""