# Import modules.
from sklearn.utils import shuffle
#from pysad.evaluation import AUROCMetric
import numpy as np

import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import time
import os 
import math

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


def stream_discord(T,w,r):
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

  #S=[i for i in S if i not in C]
  return C,S,C_score




# def discord_refinement(T,w,r): 
#   C,S=candidate_selection(T,w,r)
#   dist ={}
#   for c in C:
#     dist[str(c)] = float('inf')
#   for s in S :
#     for c in C:
#       if s==c:
#         continue
#       d=distance(T[s:s+w],T[c:c+w])#replace by early abandon
#       if (d<r):
#         C.remove(c)
#       dist[str(c)]=min(d,dist[str(c)])
#   return C
# print(discord_refinement(T,w,r))
#plot_time_series(T)

#plot_time_series(df["column1"])





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




class class_our:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")
   
    
    def test(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):

        #@jit
        
      def our(X,w,r):
        #X should be a one dimensional vector
        _,_,scores =stream_discord(X,w,r)
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
          scores= our(X,w=args["window"],r=args["threshold"])
          scores =score_to_label(nbr_anomalies,scores,gap)
          

          return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


      possible_window=np.arange(100,gap+200)
      possible_threshold=np.arange(1,10,0.5)
      space2 ={"window":hp.choice("window_index",possible_window), "threshold":hp.choice("threshold_index",possible_threshold)}
      trials = Trials()
      
      
      """best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=20,trials = trials)
      #print(best)
      start =time.monotonic()
      real_scores= our(X,w=possible_window[best["window_index"]], r=possible_threshold[best["threshold_index"]] )
      end =time.monotonic()"""
      
      start =time.monotonic()

      best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=1,trials = trials)
      #print(best)
      end =time.monotonic()
      best_param=best_param={"window":possible_window[best["window_index"]], 'threshold':possible_threshold[best["threshold_index"]] }

      return np.zeros(len(X)), np.zeros(len(X)), [],0, best_param, end-start      
      """best_param={"window":possible_window[best["window_index"]], 'threshold':possible_threshold[best["threshold_index"]] }
      
      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      #print("the final score is", scoring(scores_label),identified)
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start"""


        