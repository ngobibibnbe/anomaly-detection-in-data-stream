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


#ok

def overlapping_merlin(identified, expected,gap):
  score=0
  for expect in expected: 
    if identified in [*range (int(expect)-gap,int(expect)+gap)]:
      score+=1
  return score/len(expected)

class class_iforestASD:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
            
    def test(X,right,nbr_anomalies,gap):

        #@jit
        def iforestASD(X,window_size,n_estimators=100):
            X =np.array(X)
            """
            Malheureusement le concept drift n'est pas encore implémenté dans pysad nous devons le faire manuellement
            n_estimators est le nombre d'arbres nécéssaire
            Window_size est la taille de la fenêtre
            """
            initial=window_size
            np.random.seed(61)  # Fix random seed.
            drift_detector = drift.ADWIN()
            X_all =X
            #iterator = ArrayStreamer(shuffle=False)
            model=models.IForestASD(initial_window_X=X_all[:initial],window_size=window_size)
            model.fit(X_all[:window_size])
            model.n_estimators=100

            i=0
            scores=np.array([])
            batchs=range (0,len(X_all),window_size)
            flag=False
            for batch in range (0,len(X_all),window_size):
                i+=1
                #score =model.score(X_all[batch:batch+window_size])
                if flag==True:
                    model=models.IForestASD(initial_window_X=X_all[batch:batch+window_size],window_size=window_size)
                    model.fit(X_all[batch:batch+window_size])
                    flag=False
                #print(scores)
                score =model.score(X_all[batch:batch+window_size])
                for id,one_score in enumerate(score) :
                    in_drift, in_warning =  drift_detector.update(one_score)
                    if in_drift:
                        print("changed detected at", id+batch)
                        flag=True
                        break
                
                score =np.array(score)
                scores=np.concatenate((scores, score)) # scores+score"""
            #scores =np.zeros(len(X_all))#np.array(model.score(X_all))
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
            return score
        
        def objective(args):
            print(args)
            scores= iforestASD(X,window_size=args["window_size"],n_estimators=args["n_estimators"]
                                )
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_nbr_tree =np.arange(1,25)#[*range(1,100)]
        possible_window_size =np.arange(200,1000) #[*range(200,1000)]
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)
        , "n_estimators":hp.choice("n_estimators_index",possible_nbr_tree)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        end =time.monotonic()
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"window_size":possible_window_size[best["window_size_index"]],"n_estimator": possible_nbr_tree[best["n_estimators_index"]]}
        #print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), str(best_param), end-start







class Cluster:
  def __init__(self,subsequence,radius):
    self.radius =radius
    self.nb_clustroid=4
    self.outliers=[]
    self.clusters=[[subsequence]]
  def add_cluster(self,subsequence):
    self.clusters.append([subsequence])
def clustering(Cluster,r, subsequence) :
  dist=r
  max_dist =0
  cluster_id=False
  there_is_a_cluster =False
  # try to identify its cluster 
  for id_cluster,cluster in enumerate(Cluster.clusters):
    for clustroid in cluster:
      if z_norm_dist(clustroid,subsequence)<dist:
        dist=z_norm_dist(clustroid,subsequence)
        cluster_id =id_cluster
        # try to know if it can be the centroid
        max_dist=max(max_dist,z_norm_dist(clustroid,subsequence))
  if max_dist >r and cluster_id!=False:
    print("Rien fait: Cette partie est délicate car on essaie d'optimiser le rayon du cluster")
    # on fait un clustering hierarchique pour garder un certain rayon dans notre algorithme de clustering 
  # try to know if it can be the centroid
  if cluster_id!=False:
    if len(Cluster.clusters[cluster_id])<Cluster.nb_clustroid and not any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]) :
      Cluster.clusters[cluster_id].append(subsequence)
    elif  any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]):
      return True
    else:
      dist_matrice=np.array([ [z_norm_dist(i,j) for i in Cluster.clusters[cluster_id] ] for j in Cluster.clusters[cluster_id]])
      min_dist = dist_matrice[dist_matrice != 0].min()
      ij_min = np.where(dist_matrice == min_dist)[0]
      ij_min = tuple([i.item() for i in ij_min])
      #if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
      if dist>min_dist:
        Cluster.clusters[cluster_id][ij_min[0]]=subsequence
    return True 
  else:
    return False



class class_our_stream:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
    
    def test(X,right,nbr_anomalies,gap):
        

        def distance(a,b):
            dist = np.linalg.norm(a - b)
            #print(dist)
            x=a
            y=b
            maxi=max(np.mean(x)/np.mean(y),np.mean(y)/np.mean(x))
            return np.linalg.norm(a - b)*math.sqrt(2*len(x)*(1-np.corrcoef(x,y)[0][1]))#dist
        def z_norm_dist(x,y):
            maxi=max(np.mean(x)/np.mean(y),np.mean(y)/np.mean(x))
            return np.linalg.norm(x - y)*math.sqrt(2*len(x)*(1-np.corrcoef(x,y)[0][1]))


        def stream_discord(T,w,r):
            T=np.array(T).reshape(-1,1)
            S=[*range(0,len(T),int(w/2))]
            to_remove=[]
            for idx,s in enumerate(S) :
                if (len(T)<S[idx]+w):
                    to_remove.append(s)
                    #S.remove(s)  # to correct later
            for e in to_remove :
                    S.remove(e)  
            C=[S[0]]
            cluster=Cluster(T[S[0]:S[0]+w],r)
            C_score=np.zeros(len(T))
            C_score[S[0]]=float('inf')
            #print(C)
            for s in [i for i in S if i not in C]:
                    isCandidate=True
                    min_dist_if_discord=float('inf')
                    for c in C :
                        #print(s,"*",s+w,"**",len(T))
                        min_dist_if_discord=min(min_dist_if_discord,distance(T[s:s+w],T[c:c+w]))
                        C_score[c]=min(C_score[c],distance(T[s:s+w],T[c:c+w]))
                        if distance(T[s:s+w],T[c:c+w])< r:
                            #print([s])
                            C.remove(c)
                            C_score[c]=0
                            isCandidate=False
                            # Normalement ici aussi on devrait l'ajouter au cluster mais le clustering n'est pas encore très bon et lui donner trop de responsabilité peut être difficile
                    
                    if isCandidate and not clustering(cluster,r,T[s:s+w]):
                        C.append(s)
                        C_score[s]=min_dist_if_discord
                    if not isCandidate and not clustering(cluster,r,T[s:s+w]):
                        cluster.add_cluster(T[s:s+w])

           
            return C,S,C_score
        #@jit
        
                
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
            scores= stream_discord(X,window_size=args["window_size"],r=args["threshold"]
                                ) # ajouter l'exclusion et le rayon de la formule de clustering
            scores =score_to_label(nbr_anomalies,scores,gap)
            
            return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


        possible_nbr_tree =np.arange(1,25)#[*range(1,100)]
        possible_window_size =np.arange(200,1000) #[*range(200,1000)]
        space2 ={"window_size":hp.choice("window_size_index",possible_window_size)
        , "n_estimators":hp.choice("n_estimators_index",possible_nbr_tree)}
        trials = Trials()
        best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
        #print(best)
        start =time.monotonic()
        real_scores= iforestASD(X,window_size=possible_window_size[best["window_size_index"]],n_estimators=possible_nbr_tree[best["n_estimators_index"]])
        end =time.monotonic()
        scores_label =score_to_label(nbr_anomalies,real_scores,gap)
        identified =[key for key, val in enumerate(scores_label) if val in [1]] 
        best_param={"window_size":possible_window_size[best["window_size_index"]],"n_estimator": possible_nbr_tree[best["n_estimators_index"]]}
        #print("the final score is", scoring(scores_label),identified)
        return real_scores, scores_label, identified,scoring(scores_label), str(best_param), end-start





# Test pipeline   les threshold des methodes coe iforest seront récupérés dans NAB parce qu'NAB à une fonction de score automatisé. 

#@jit                   
def test () :                                                         
    base_file ='Pattern_lengths_and_Number_of_discords2.xlsx'
    base = pd.read_excel(base_file)
    methods= {"iforestASD":class_iforestASD}# "iforestASD_SUB":iforestASD_SUB,"subSequenceiforestASD":iforestASD } #"iforestASD":iforestASD, "HStree":HStree "MILOF":MILOF
    
        
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
            gap =int(base["discord length"][idx]) + int(int(base["Dataset length"][idx])/100)
            nbr_anomalies=len(str(base["Position discord"][idx]).split(';'))


            if key =="iforestASD":
                real_scores, scores_label, identified,score,best_params[idx], time_taken[idx]= class_iforestASD.test(X,right,nbr_anomalies,gap)  # Le concept drift est encore à faire manuellement et;le threshold est fixé après en fonction du nombre d'anomalies dans le dataset pour ne pas pénaliser l'algorithme
            if key =="iforestASD_SUB":
                scores = iforestASD_SUB(X,500)
            if  key=="MILOF":# and "art_load_balancer_spikes.csv" in dataset:
                if len(np.unique(np.array(X),axis=0))<500:
                    print(len(X))
                    scores =[0.0 for i in X]
                else:
                    scores = method(X)
            if key=="HStree":
                scores = method(X)
            #calling methods
 
 
 
            
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

