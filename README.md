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
1. [Methods compared](#Emotional-Models): Presentation of methods we compared
2. [Datasets and their caracteristics](#Emotional-Models): Brief Description of datasets and caracteristics identified 
3. [Description of the experimental protocol](#Emotional-Models): Description of the experimental protocol
5. [Results](#Emotional-Models): Presentation of results obtained
6. [Reproducibility](#Emotional-Models): How to reproduce our tests
7. [Referencies](#Emotional-Models)





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
