# anomaly-detection-in-data-stream

#to perform the test :
pip install -r requirements.txt

  #For testing univariate dataset
  python test_univariate.py name-of-the-method-to-test

  #For testing multivariate datasets 
  python test_multivariate.py name-of-the-method-to-test

  the name of methods are the following:

  MILOF for MILOF
  ARIMAFD for online ARIMA
  HS-tree for Hs-tree
  iforestASD for iForestASD
  KitNet for KitNet

The results of the test will be in the folder result. for each dataset, the result file contains:
-the time execution
-the f1-score
-the best hyperparameters of each method


Details on characteristics of the datasets and hyperparameters we found are summarized in the file: summary.pdf. 
