# Benchmarking data stream outlier detection methods

### Main Contributions
Data stream datasets have characteristics depending on the underlying domain and context. From their proximity to time series (mmmmmmmmmm citer le papier), we can characterize a data stream by the presence of  seasonality, trend, and cycle; due to the way data are arriving, we can add concept drift which is a non-negligible phenomenon which currently occurs in data stream context. 

This work: 

:white_check_mark: Compare some data stream anomaly detection methods on their latences and performances 

:white_check_mark: Focus on caracteristics presents on the datasets (seasonality, trend, cycle, concept drift) 

### Interested in my work?

Feel free to contact me at: anne.ngobibinbe@gmail.com

*The final version of our paper (in French) on the benchmark of data stream outlier detection methods is being submitted to the 2022 French Speaking Conference on the Extraction and Management of Knowledge (EGC).*

### README Structure
1. [Methods compared](#Methods-compared): Presentation of methods we compared
2. [Datasets and their caracteristics](#Datasets-and-their-caracteristics): Brief Description of datasets and caracteristics identified 
3. [Description of the experimental protocol](#Description-of-the-experimental-protocol): Description of the experimental protocol
5. [Results](#Results): Presentation of results obtained
6. [Reproducibility](#Reproducibility): How to reproduce our tests
7. [Referencies](#Referencies)


## Methods compared
As it's the case for most of the anomaly detection methods, the following methods produce an anomaly score for each incoming instance showing how well the instance could be an anomaly, finally a threshold fixed by the user permits to say that instances with anomaly scores higher than the threshold are anomalies. In the literature, data stream anomaly detection methods are mostly separated into statistical based, tree based, proximity based and deep learning based approaches. We have chosen highly used and recommended approaches in each of those categories. 

Methods:
1. Online ARIMA : Statistic based methods which provide the anomaly score by computing the distance between the value of the instance forecasted from past instances and the real value of the instance. (mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm rajouter son lien et ceux des autres, mettre aussi le lien directe vers le github)
2. HStree : Tree based approach, providing the anomaly score according to how well an instance is isolated from other instances in an ensemble of pre-constructed trees
3. IforestASD : Similar to HStree
4. KitNet : Deep learning based methods  providing the anomaly score as the reconstruction error of an instance (Autoencoder)
5. MILOF : Proximity based approach, providing the anomaly score according to how locally reachable is an instance compared to its nearest neighbours. 

## Datasets and their caracteristics
We selected datasets mostly from IOT domain and whose anomalies causes are known to avoid errors due to human or tools labeling. In the boards, **no** trend means the dataset has a constant trend. Those characteristics have been identified by visualizing the datasets and are support by STL decompositions for trends and seasonalities.

:link: Anchor Links:
1. [Univariate datasets](#Univariate-datasets)
2. [Multivariate datasets](#Multivariate-datasets)

### Univariate datasets
We used here the Real known cause group of datasets of the NAB Benchmark (mmmmmmmmmmmmmmmmmmmmciter nab, mettre aussi le lien directe vers le github)

Dataset | Domain | Dataset length | number of anomalies | Concept Drift | Seasonality | Trend | Cylce 
-----|-------------|------------|------------|---------|-----------------------|------------ |------------
ambiant temperature system failure|  |7267 | 2| yes | yes | yes| no
cpu utilization asg misconfiguration|  |18050 | 1| yes | yes | yes| yes
ec2 request latency system failure|  |4032 | 3| no | no | yes| no
machine temperature system failure|  |22695 | 4| no | no | no| no
new york taxi|  |10320 | 5| no | yes | yes| yes
rogue agent keyhold|  |1882 | 2| yes | no | yes| no
rogue agent key up down|  |5315 | 2| yes | no | no| no

 
### Multivariate datasets
We selected some datasets showing a great number of our specified characteristics from the SKAB benchmark (mmmmmmmmmmciter SKAB). All those datasets have 7 dimensions.


Dataset | Domain | Dataset length | number of anomalies | Concept Drift | Seasonality | Trend | Cylce 
-----|-------------|------------|------------|---------|-----------------------|------------ |------------
other 9: Closing the valve at the flow inlet to the pump| IOT |751 | 2| no | no | yes| yes
other 11: Closing the valve at the flow inlet to the pump| IOT |665 | 4| no | yes | no| no
other 13: Sharply behavior of rotor imbalance| IOT |7267 | 2| yes | yes | yes| no
other 14: Linear behavior of rotor imbalance| IOT |1153 | 2| yes | yes | yes| yes
other 15: Step behavior of rotor imabalance |  IOT|1147 | 2| yes | yes | yes| no
other 17: Exponential behavior of rotor imbalance| IOT |1147 | 4| no | yes | no| yes
other 20: Draining water from the tank until cavation | IOT |1191 | 4| yes | yes | yes| no
other 22: Water supply of increased temperature | IOT |1079 | 4| yes | yes | yes| yes


## Description of the experimental protocol
For each dataset, a bayesian optimization is performed to find best hyperparameters (details of the hyperparameter search space of each method could be found in the implementation details (page 8) section of the [summary_of_the_experiment](https://github.com/nams2000/anomaly-detection-in-data-stream/blob/master/summary_of_the_experiments.pdf) file), then we test the method with the best hyperparameters and record the execution time and the f1-score. Finally we process the latence or response time (average time to treat an instance) (**latence =the execution time on the dataset/length of the dataset**). To process the f1-score, we consider a method find an anomaly if it 1% of the length of the dataset around the position of the anomaly (this because an anomaly generaly occurs on a small period and the point given as the position of the anomaly is a point inside the period on which the anomaly occured).

## Results
Due to conception restrictions KitNet couldn't be applied on univariate datasets and Online ARIMA can't be applied on multivariate datasets. 

:link: Anchor Links:
1. [Results on univariate datasets](#Results-on-univariate-datasets)
2. [Results on multivariate datasets](#Results-on-multivariate-datasets)

### Results on univariate datasets
1. F1-score 

Dataset | MILOF | IforestASD | HStree | Online ARIMA 
-----|-------------|------------|------------|-----------
ambiant temperature system failure| 0.4| **0.67** | 0.3 | **0.67**
cpu utilization asg misconfiguration|  0.5 |0.42 | 0.45 | **1**
ec2 request latency system failure| 0.5 | 0.343 | **0.94**  | 0.8 
machine temperature system failure| 0.15 | 0.7825 |**0.88** | 0.66
new york taxi|0.25 | 0.31 | 0.5 | **0.6**
rogue agent keyhold|  0.136 | 0.33 | 0.079 | 0.1
rogue agent key up down| 0.4 | **0.67** | 0.15 | 0.11


2. Execution time (ms)

 Dataset | MILOF | IforestASD | HStree | Online ARIMA 
-----|-------------|------------|------------|-----------
ambiant temperature system failure| 172 | 200 | 212 | **50**
cpu utilization asg misconfiguration| 430 | 438 | 738 | **129**
ec2 request latency system failure| 51 | 167 | 125 | **38**
machine temperature system failure| 560 | 580 | 9752 | **109**
new york taxi | **275** | 269 | 4776 | 391
rogue agent keyhold|  31 | 76| **16** | 17
rogue agent key up down| 26  | 203 | **8** | 37


 
### Results on multivariate datasets
1. F1-score 

Dataset | Domain | Dataset length | number of anomalies | Concept Drift | Seasonality | Trend | Cylce 
-----|-------------|------------|------------|---------|-----------------------|------------ |------------
other 9: Closing the valve at the flow inlet to the pump| IOT |751 | 2| no | no | yes| yes
other 11: Closing the valve at the flow inlet to the pump| IOT |665 | 4| no | yes | no| no
other 13: Sharply behavior of rotor imbalance| IOT |7267 | 2| yes | yes | yes| no
other 14: Linear behavior of rotor imbalance| IOT |1153 | 2| yes | yes | yes| yes
other 15: Step behavior of rotor imabalance |  IOT|1147 | 2| yes | yes | yes| no
other 17: Exponential behavior of rotor imbalance| IOT |1147 | 4| no | yes | no| yes
other 20: Draining water from the tank until cavation | IOT |1191 | 4| yes | yes | yes| no
other 22: Water supply of increased temperature | IOT |1079 | 4| yes | yes | yes| yes

2. Execution time 



## How to reproduce our tests 
### Install dependencies:
Make sure you have at least python 3.6 
to install requirement type: pip install -r requirements.txt

### To test methods:
On univariate dataset:
python test_univariate.py name-of-the-method-to-test

On multivariate datasets: 
python test_multivariate.py name-of-the-method-to-test

The name of methods are the following:
  MILOF for MILOF,  ARIMAFD for online ARIMA,  HS-tree for Hs-tree, iforestASD for iForestASD, KitNet for KitNet,

The results of the test will be in the folder result. for each dataset, the result file contains:
-the time execution

-the f1-score

-the best hyperparameters of each method

# Notices: 
It is possible to change the score used for the experiment by default the MERLIN score (1% around the anomaly )is used.

Details on characteristics of the datasets and hyperparameters we found are summarized in the file: summary_of_the_experiment.pdf. 


# referencies:
- 
