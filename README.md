# anomaly-detection-in-data-stream
Make sure you have at least python 3.6 
# to install dependencies:
pip install -r requirements.txt

# For testing univariate dataset:
python test_univariate.py name-of-the-method-to-test

#For testing multivariate datasets: 
python test_multivariate.py name-of-the-method-to-test

# The name of methods are the following:
  MILOF for MILOF
  ARIMAFD for online ARIMA
  HS-tree for Hs-tree
  iforestASD for iForestASD
  KitNet for KitNet

# The results of the test will be in the folder result. for each dataset, the result file contains:

-the time execution
-the f1-score
-the best hyperparameters of each method

# It is possible to change the score used for the experiment by default the MERLIN score (1% around the anomaly )is used.

# Details on characteristics of the datasets and hyperparameters we found are summarized in the file: summary_of_the_experiment.pdf. 
