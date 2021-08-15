# transforming nab dataset and adding it to our dataset form
# deplacer ce fichier à la racine 
import time
import json
import os 
from datetime import datetime

base='nab_label.json'
#methods= {"hotsax":hotsax} #"Matrix_profile":Matrix_profile,"hotsax":hotsax,"deepant_execution":deepant_execution, "MERLIN":MERLIN
fileObject = open(base, "r")
jsonContent = fileObject.read()
aList = json.loads(jsonContent)
base_file ='Pattern_lengths_and_Number_of_discords.xlsx'
base_notes = pd.read_excel(base_file)
for dataset, anomalies in aList.items():
  print(dataset)
  df = pd.read_csv("real_nab_data/nab-data/"+dataset)
  column="value"
  df["index"]=df.index
  df["timestamp"]=pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
  directory = os.path.dirname(("dataset/nab-data/"+dataset))
  if not os.path.exists(directory):
        os.makedirs(directory)
  df["value"].to_csv("dataset/nab-data/"+dataset)
  if anomalies!=[]:
    index_anomalies=[]
    max_window=0
    for anomaly in anomalies:
      index1= df[df['timestamp']==datetime.fromisoformat(anomaly[0])].index.values[0] 
      window= df[df['timestamp']==anomaly[1]].index.values[0]  - df[df['timestamp']==anomaly[0]].index.values[0] 
      max_window=max(window, max_window)
      index_anomalies.append(index1)
    index_anomalies=str(index_anomalies).strip('[]').replace(",",";")

    #********THIS PART IS FOR ABNORMAL POINT *********#
    #index_anomalies=[df[df['timestamp']==ano].index.values[0] for ano in anomalies]
    #index_anomalies=str(index_anomalies).strip('[]').replace(",",";")
    #********THIS IS IMPORTANT *******************

    new_data=pd.DataFrame({"discord length":max_window,"Dataset length":len(df["value"].values),"Dataset":["nab-data/"+dataset],"Type":["NAB"],"#discords":[len(anomalies)],"Position discord":[index_anomalies]})
    base_notes=base_notes.append(new_data, ignore_index = True)
  
base_notes.to_excel("Pattern_lengths_and_Number_of_discords2.xlsx")


# # #datasets : 
# # UCI : https://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/ tiré de https://arxiv.org/pdf/1908.01146.pdf

# # points anormaux :  https://www.researchgate.net/publication/340374826_Anomaly_Detection_in_Univariate_Time-series_A_Survey_on_the_State-of-the-Art
# # the Robust Random Cut Forest: https://klabum.github.io/rrcf/taxi.html,  

# # peut êêtre: https://arxiv.org/pdf/1703.09752.pdf https://project.inria.fr/aaltd19/files/2019/08/AALTD_19_Karadayi.pdf  faster hotsax: https://arxiv.org/abs/2101.10698