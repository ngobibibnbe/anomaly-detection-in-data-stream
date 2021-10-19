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
from datetime import datetime

def distance(a,b):
  x=a
  y=b
  return np.linalg.norm(a - b)*math.sqrt(2*len(x)*(1-np.corrcoef(x,y)[0][1])) #******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée
def z_norm_dist(x,y):
  return np.linalg.norm(x - y)*math.sqrt(2*len(x)*(1-np.corrcoef(x,y)[0][1]))  #******songer à tester sans la distance normale ajoutée  pour voir l plus de la distance proposée


class Cluster:
  def __init__(self,subsequence,radius,max_clusters):
    self.radius =radius
    self.nb_clustroid=4
    self.outliers=[]
    self.clusters=[[subsequence]]
    self.clusters_activity=[datetime.now()]
    self.max_clusters=max_clusters
  def add_cluster(self,subsequence):
    if len(self.clusters) >self.max_clusters:
      min_index = self.clusters_activity.index( min(self.clusters_activity))
      print("*************** This is the cluster that had the lowest activity",min_index, self.clusters_activity)
      self.clusters_activity.pop(min_index)
      self.clusters.pop(min_index)

    self.clusters_activity.append(datetime.now())
    self.clusters.append([subsequence])
def clustering(Cluster,r, subsequence) :
  dist=r
  min_dist =float('inf')
  cluster_id=False
  there_is_a_cluster =False
  # try to identify its cluster 
  for id_cluster,cluster in enumerate(Cluster.clusters):
    for clustroid in cluster:
      if z_norm_dist(clustroid,subsequence)<dist:
        print("********************************, it entered a cluster")
        dist=z_norm_dist(clustroid,subsequence)
        if z_norm_dist(clustroid,subsequence)<min_dist:
          min_dist = z_norm_dist(clustroid,subsequence)
          cluster_id =id_cluster
        # try to know if it can be the centroid
  """if min_dist >r and cluster_id!=False:
    print("Rien fait: Cette partie est délicate car on essaie d'optimiser le rayon du cluster")
    # on fait un clustering hierarchique pour garder un certain rayon dans notre algorithme de clustering 
  """
  # try to know if it can be the centroid
  if cluster_id!=False:
    Cluster.clusters_activity[cluster_id]=datetime.now()

    print("********************Ajout à un cluster",cluster_id, np.array(Cluster.clusters).shape)
    if len(Cluster.clusters[cluster_id])< Cluster.nb_clustroid and not any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]) :
      Cluster.clusters[cluster_id].append(subsequence)
      
    elif  any(np.array_equal(subsequence, i) for i in Cluster.clusters[cluster_id]):
      #print("**** il avait deja son jumeau")
      return True
    else:
      #print("******************** il vient remplacer un clustroid")
      #if there is a subsequence which is too near another subsequence, we can remove it . we can do it because l'inégalité triangulaire est vérifiée
      dist_matrice=np.array([ [z_norm_dist(i,j) for i in Cluster.clusters[cluster_id] ] for j in Cluster.clusters[cluster_id]])
      min_dist = dist_matrice[dist_matrice != 0].min()
      ij_min = np.where(dist_matrice == min_dist)[0]
      ij_min = tuple([i.item() for i in ij_min])
      if dist>min_dist:
        Cluster.clusters[cluster_id][ij_min[0]]=subsequence
    return True 
  else:
    return False


def stream_discord(T,w,r,training,max_clusters):
  S=[*range(0,len(T),int(w/2))]
  to_remove=[]
  for idx,s in enumerate(S) :
    if (len(T)<S[idx]+w):
      to_remove.append(s)
      #S.remove(s)  # to correct later
  for e in to_remove :
    S.remove(e)  
  C=[S[0]]
  cluster=Cluster(T[S[0]:S[0]+w],r,max_clusters)
  C_score=np.zeros(len(T))
  C_score[S[0]]=float('inf')
  #print(C)
  for s in [i for i in S if i not in C]:
    isCandidate=True
    min_dist_if_discord=float('inf')
    for c in C :
      #print(s,"*",s+w,"**",len(T))
      min_dist_if_discord=min(min_dist_if_discord,distance(T[s:s+w],T[c:c+w]))
      if c<= training: ##********because we can't update at every time
        C_score[c]=min(C_score[c],distance(T[s:s+w],T[c:c+w]))
      if distance(T[s:s+w],T[c:c+w])< r:
        C.remove(c)
        if not clustering(cluster,r,T[c:c+w]):
          cluster.add_cluster(T[s:s+w])
        if c<= training: #*********because we can't update at every time
          C_score[c]=0 #******** voir comment ajouter un temps d'attente 
        isCandidate=False
        # Normalement ici aussi on devrait l'ajouter au cluster mais le clustering n'est pas encore très bon et lui donner trop de responsabilité peut être difficile
        
    if isCandidate and not clustering(cluster,r,T[s:s+w]):
      C.append(s)
      C_score[s]=min_dist_if_discord
    if not isCandidate and not clustering(cluster,r,T[s:s+w]):
      print("***************************it's not entering any cluster")
      cluster.add_cluster(T[s:s+w])

  #S=[i for i in S if i not in C]
  return C,S,C_score, cluster




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


def check (indice, real_indices,gap):
    Flag=True
    for real_indice in real_indices:
        #print(indice, [*range(real_indice-gap,real_indice+gap)])
        search = np.arange(real_indice,real_indice+gap)
        if indice in search:
            Flag=False
    return Flag

"""def score_to_label(nbr_anomalies,scores,gap):
  
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
        real_indices = np.append(real_indices,indice)
        #print("**",threshold,(real_indices))
    label =np.where(scores>=threshold)
    score_label =np.zeros(len(scores))
    for i in label :
      score_label[i]=1
      
  return score_label # [0 if i<threshold else 1 for i in scores ]


"""

class class_our:
    def __init__(self):
        #self.nbr_anomalies= nbr_anomalies
        print("ok")   
    def test(dataset,X,right,nbr_anomalies,gap,scoring_metric="merlin"):
      def our(X,w,r,training,cluster):
        #X should be a one dimensional vector
        _,_,scores,clust =stream_discord(X,w,r,training,cluster)
        print("*********nbr of clusters",np.array(clust.clusters).shape)
        return scores

                        
        #right=[387,948,1485]
        #nbr_anomalies=3
        
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
          scores= our(X,w=args["window"],r=args["threshold"],training=args["training"],cluster=args["cluster"])
          scores =score_to_label(nbr_anomalies,scores,gap)          
          return 1/(1+scoring(scores))#scoring(scores)#{'loss': 1/1+score, 'status': STATUS_OK}


      possible_window=np.array([gap,gap])#arange(100,gap+200)
      possible_threshold=np.arange(1,10,0.5)
      right_discord =[ int(discord) for discord in right]
      possible_training=np.arange(1, min(min(right_discord),int(len(X)/4)))
      possible_cluster=np.arange(10, 30)
      space2 ={"training":hp.choice("training_index",possible_training),
      "window":hp.choice("window_index",possible_window), "threshold":hp.choice("threshold_index",possible_threshold),
       "cluster":hp.choice("cluster_index",possible_cluster)}
      trials = Trials()
      
      
      """best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=30,trials = trials)
      #print(best)
      start =time.monotonic()
      real_scores= our(X,w=possible_window[best["window_index"]], r=possible_threshold[best["threshold_index"]]
      , training=possible_training[best["training_index"]], cluster=possible_cluster[best["cluster_index"]] )
      end =time.monotonic()

      best_param={"cluster":possible_cluster[best["cluster_index"]], "training":possible_training[best["training_index"]],"window":possible_window[best["window_index"]], 'threshold':possible_threshold[best["threshold_index"]] }
      scores_label =score_to_label(nbr_anomalies,real_scores,gap)
      identified =[key for key, val in enumerate(scores_label) if val in [1]] 
      #print("the final score is", scoring(scores_label),identified)
      print("*********identified",identified)
      return real_scores, scores_label, identified,scoring(scores_label), best_param, end-start"""
      
      start =time.monotonic()

      best = fmin(fn=objective,space=space2, algo=tpe.suggest, max_evals=1,trials = trials)
      #print(best)
      end =time.monotonic()
      best_param={"cluster":possible_cluster[best["cluster_index"]], "training":possible_training[best["training_index"]],"window":possible_window[best["window_index"]], 'threshold':possible_threshold[best["threshold_index"]] }
      
      return np.zeros(len(X)), np.zeros(len(X)), [],0, best_param, end-start     

        