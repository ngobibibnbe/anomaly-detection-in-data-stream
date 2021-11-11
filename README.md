# Dependencies
Make sure you have at least python 3.6 
to install requirement type: pip install -r requirements.txt

# To test methods:
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






Methodology and implementation {#cp:chapitre2}
==============================

Datasets
--------

In this section we will describe the datasets That we used in our
experiment and provide their characteristics. We will also describe the
evaluation metric, and the hyperparameter spaces we used.

### Univariate dataset

There are various existing labeled dataset that could be extracted from
the Yahoo benchmark, the Numenta benchmark, the UCR benchmark and many
others. Unfortunately Yahoo wasn’t opened and we made a demand, but it
has been rejected. UCR too wasn’t available when we made our tests. So
we focused on Numenta which is almost sufficient, and particularly
interesting since it’s highly used in the literature. Numenta provide 52
labeled datasets in 7 groups:

-   **Datasets with artificial anomalies :** in this set of datasets,
    the complete streams present seasonalities (for some of them), with
    noises, but none of those datasets present trends.

-   **Datasets extracted by the Amazon CloudWatch service :** those
    datasets record CPU utilization, network bytes, and disk read bytes.
    When visualizing the datasets, we remark that they present : cycle,
    seasonality, noises, drift and no trend.

-   **Datasets of online advertisement clicking rate :** these datasets
    record the cost-per-click and the cost-per-thousand impression on
    clicking rate. The datasets also present seasonalities and trends,
    cycles, and noises.

-   **Datasets on real known causes : ** here, the causes were known and
    labeled as the event occurred. Those are mainly on known systems
    failure such as the ambient temperature or machine temperature
    failure, but there are also datasets on fraud detection. Here
    datasets are mostly showing cycles, noises.

-   **Datasets on real traffic data from Minnesota :** those datasets
    are mostly showing noises and cycles.

    **Datasets on real twitter mentions on a publicly traded company
    such as google or IBM :** the metric value represents the number of
    mentions for a given stock symbol every 5 minutes. Here datasets are
    showing noises and cycles.

The datasets extracted from **real known causes** are interesting since
they don’t depend on NAB appreciation. And they permit a diverse form of
data stream. **Amazon cloud watch datasets ** and **Real advertisement
clicking rate** are also interesting since they present various concept
drift, and various components which can enable relevant conclusion from
our study. Those are summarized in table [table:uni-datasets].

<span>|p<span>3cm</span>|p<span>2cm</span>|p<span>1.5cm</span>|p<span>1.8cm</span>|p<span>1cm</span>|p<span>1.5cm</span>|p<span>1.5cm</span>|p<span>1.2cm</span>|</span>
[table:uni-datasets]

**Dataset** & **Domain** & **Dataset length**& **number of anomalies** &
**Drift**& **Seasonal-ity** &**Trend** & **Cycle**\
Real AWS Cloud watch on cpu utilization : 24ae8d & cpu utilization of
AWS &4032&2& no & no & constant & yes\
Real AWS Cloud watch on cpu utilization : 53ea38 & cpu utilization of
AWS & 4032&2&no & no & constant & yes\
Real AWS Cloud watch on cpu utilization : 5f5533 & cpu utilization of
AWS &4032&2& yes & no & constant & no\
Real AWS Cloud watch on cpu utilization : 77c1ca & cpu utilization of
AWS &4032&1&no &no &constant & no\
Real AWS Cloud watch on cpu utilization : 825cc2 & cpu utilization of
AWS &4032&1& no &no &constant & no\
Real AWS Cloud watch on cpu utilization : ac20cd & cpu utilization of
AWS &4032&1&yes &no &constant & no\
Real AWS Cloud watch on cpu utilization : fe7f93 & cpu utilization of
AWS &4032&3&yes &no &constant &no\

Real AWS Cloud watch on disk written bytes : 1ef3de & disk utilization
of AWS &4730&1&yes &no &constant &no\
Real AWS Cloud watch on disk written bytes : c0d644 & disk utilization
of AWS &4032&3&yes &no &constant &no\
Real AWS Cloud watch on network traffic : 257a54 & network traffic of
AWS &4032&1&yes &no &constant &no\
Real AWS Cloud watch on network traffic : 5abac7 & network traffic of
AWS &4730&2&yes &no &constant &no\

Real AWS Cloud watch on number of elb request : 8c0756 & number of elb
request of AWS &4032&2&yes &no &constant &no\

Real AWS Cloud watch on grok asg anomaly : 8c0756 & grok asg anomaly of
AWS &4621&3&yes &no &constant &no\

Real AWS Cloud watch on iio\_us-east-1\_i-a2eb1cd9\_ NetworkIn & network
traffic of AWS &1243&2&yes &yes &constant &no\

Real AWS Cloud watch on rds\_cpu \_utilization\_ cc0c53 & cpu
utilization of AWS &4032&2& yes & no &constant &no\
Real AWS Cloud watch on rds\_cpu\_ utilization\_e47b3b & cpu utilization
of AWS &4032&2& yes & no &constant &no\

Real advertisement exchange : exchange-2\_cpc\_results & Real
advertisement exchange&1621&1 & yes & yes &yes &no\
Real advertisement exchange : exchange-2\_cpm\_results & Real
advertisement exchange &1621&2& yes & yes &yes &no\
Real advertisement exchange : exchange-3\_cpc\_results & Real
advertisement exchange &1538&3& yes & yes &yes &no\
Real advertisement exchange : exchange-3\_cpm\_results & Real
advertisement exchange &1538&1& yes & yes & yes &no\

Real advertisement exchange : exchange-4\_cpc\_results & Real
advertisement exchange &1643&3& yes & yes & yes &no\

Real advertisement exchange : exchange-4\_cpm\_results & Real
advertisement exchange &1643&4& yes & yes & yes &no\

Real known cause :ambient temperature failure & Real known cause
&7267&2& yes & yes & yes &no\
Real known cause :cpu\_utilization\_asg\_ misconfiguration & Real known
cause &18050&1& yes & yes & yes &yes\

Real known cause :ec2\_request\_ latency\_ system\_failure & Real known
cause &4032&3& no & no & yes &no\
Real known cause :machine temperature system failure & Real Real known
cause &22695&4& no & no & no &no\
Real known cause :new York taxi & Real known cause &10320&5& no & yes &
yes &yes\
Real known cause :rogue agent key hold & Real known cause &1882&2& yes &
no & yes &no\

Real known cause :rogue agent key updown & Real known cause &5315&2& yes
& no & constant &no\

### Multivariate dataset

In the literature we found various available datasets of real life
problems, we mainly focused on some of them as following :

-   In the domain of fraud detection, the dataset **credit card
    detection** contains transactions made by credit cards in September
    2013 by European cardholders. This dataset shows the transactions
    that took place in two days, where we have 492 frauds out of 284,807
    transactions. The dataset is very imbalanced, the positive class
    (frauds) accounts for 0.172% of all transactions. The original
    features are not provided due to privacy policy; and given
    dimensions are results of PCA applied on original features
    @creditcard.

-   In the IOT domain, SKAB proposes a benchmark of datasets extracted
    from an industrial testbed. Those datasets were extracted from
    experiments made by closing the valve at the outlet of the flow from
    the pump, by closing the valve at the flow inlet to the pump. Others
    from data obtained from other experiments:

    -   Sharply behavior of rotor imbalance

    -   Linear behavior of rotor imbalance

    -   Step behavior of rotor imbalance

    -   Two-phase flow supply to the pump inlet

    -   Draining water from the tank until cavitation

    -   etc @skab1.

We summarized those datasets in the following table.

<span>|p<span>1.1cm</span>|p<span>1.5cm</span>|p<span>1.1cm</span>|p<span>1cm</span>|p<span>1.2cm</span>|p<span>3cm</span>|p<span>1cm</span>|p<span>1cm</span>|p<span>1cm</span>|p<span>1cm</span>|</span>
[table:datasets-multivariate-point]

**Dataset** & **Domain** & **Dataset length**& **number of anomalies** &
**Number of dimensions**& **Short description** & **Drift**&
**Season-ality** &**Trend** & **Cycle**\
other 13 & Sharply behavior of rotor imbalance &968&4& 7 & 3 dimensions
have seasonality, each of different length. Anomalies are hard to
detect, but it seems that each anomaly is related to a specific
dimension. Nevertheless, it’s difficult to be precise on the anomaly
position.&no&yes&no&no\
other 17& Exponen-tial behavior of rotor imbalance &1147&4& 7 & The
dataset presents a dimension with seasonality and the others with
cycles. &no&yes&no&yes\
other 20& Draining water from the tank until cavitation &1191&4&7& one
dimension presents a concept drift, two others with season and trends,
and the rest seems to be random fluctuation.&yes&yes&yes&no\

other 9 & Closing the valve at the flow inlet to the pump &751&2& 7 &
Two dimensions show non-constant trend with cycles.3 others show only
cycles and the two remaining are constants with some random
fluctuations.&no&no&yes&yes\
other 14& Linear behavior of rotor imbalance &1153&2& 7& One dimension
shows a concept drift from a constant trend to a trend with seasonality.
2 of the dimensions point out roughly one of the anomalies. There is one
dimension with seasonality, 2 constant ones and the rest with trends and
cycles.&yes&yes&yes&yes\

other 11& Data obtained from the experiments with closing the valve at
the flow inlet to the pump&665&4&7& 3 constant trend dimensions roughly
pointing out one of the anomalies. Another anomaly is a peak on one
dimension with seasonality, and the rest of the anomalies are not easy
to identify.&no&yes&no&no\
other 22& Water supply of increased temperature &1079&4&7& There is one
dimension with concept drift, two others with trend and seasonalities.
Another with strong long period cycles, and another with seasonality and
cycles on a constant trend; and two with a constant value and
peaks.&yes&yes&yes&yes\

other 15& Step behavior of rotor imbalance &1147&2& 7 & One dimension
with a concept drift, one with seasonality and trend; and the others are
difficult to categorize.&yes&yes&yes&no\

Evaluation metrics
------------------

In the literature various metrics have been used to assess anomaly
detection methods in streaming data. In the frame of point anomaly
detection, classic metrics like the AUC, precision, recall,f1 score has
been used @Al-amri2021.

But other methods rather used a metric giving the credit to the method
when it’s at + or - 1% near the abnormal point @Nakamura2020. We think
it is more relevant to check it.

Other’s estimated that it’s difficult to provide accurately the exact
position of the anomaly in various cases. When working with streaming
data, detecting anomalies as soon as possible is worthy. In this mood,
NAB proposes a score which aims to reward anomalies detected around the
ground truth label but also early detection. For constructing the
neighbourhood around the ground truth, NAB authors propose to share 10%
of the time series length among the existing anomalies in a data file.
The length attributed to each ground truth anomaly will represent the
length of the window around it. For this NAB consider an application
profile **A** which is a matrix having weights for **false positive
($A_{FP}$)** , **true positive ($A_{TP}$)**, **true negative
($A_{TN}$)** and **false negative detection ($A_{FN}$)**. The reason for
this matrix is the fact that for some domains, **false positive** labels
have to be more penalized than **false negative** and the inverse for
other domains. So the application profile permits to weigh the way the
algorithm will be rewarded and penalized. By considering a given window
around the ground truth label, an anomaly detected inside this window
and **y** the relative position of the detected anomaly. NAB scores this
detection by equation [eq : nab]. The final score for a data file is
given by equation [eq : nab2].The score of all data files is given by
[eq : nab32] and the normalized score is given by [eq : nab3].

$$\sigma^A(y)= (A_{TP}-A_{FP})(1/(1+e^{5y})) -1
    \label{eq : nab}$$

$$S^A_d= (\sum_{y \in Y_d} \sigma^A(y)) + A_{FN}f_d
    \label{eq : nab2}$$

$$S^A_d= \sum_{d \in D} S^A_d 
    \label{eq : nab32}$$

$$\label{eq : nab3}
    S^A_{NAB}= 100* (S^A -S^A_{null})/ (S^A_{perfect}-S^A_{null})$$

@Lavin2015

An example of NAB scoring procedure could be find at [fig:nab]

![Example of NAB scoring
procedure.](figures/figure_sophie/nab_scoring.PNG "fig:")<span>Example
of NAB scoring procedure. In a given window, only earlier detection is
valorized. </span> [fig:nab]

For **point anomaly detection**, the score of MERLIN and NAB are two
interesting scores. What is more interesting is the zone around the
anomaly given by NAB. NAB defined this score because in many more cases
the anomaly is difficult to point out, we can even think the anomaly is
a subsequence one. This impacted on the quality of labels given for the
exact position of anomalies of much datasets used for abnormal point
detection; those problems had been more highlighted in @wu_current_2020.
There are also many papers complaining on the exact position of
anomalies given for NAB datasets. For these reasons we decided to **give
the full point to a method if it finds an anomaly in the abnormal region
provided for each dataset of NAB** as in @hexagon-ml and in MERLIN. For
each dataset we only have anomaly score on each point of the dataset,
anomalies position with a gap (representing the abnormal region) that we
defined as 1% of the time series length (MERLIN) . We processed the f1
score on all possible thresholds on the anomaly scores provided by the
method. And finally kept the threshold providing the higher f1 score. A
key element here is how we process the precision and the recall for the
f1 score. For the recall we did as NAB authors, the true positive is
given by the number of regions found (A region is found if the model
identify an anomaly at at least one point in the abnormal region); and
finally the recall is computed as the rate of abnormal region found by
the method among all abnormal regions. We computed the precision as the
rate of points pointing to a real abnormal region among all points
detected. Finally the f1 score =
$\frac{(2*recall*precision)}{recall + precision}$.

In the next section we will briefly describe how we implemented all
methods and give details of test procedures.

Implementation details
----------------------

In this section we will provide the libraries used for the
implementation of each method, and the search space provided for
hyperparameters tuning procedure.

The language used for the tests is Python, and the library used for
hyperparameters tuning is hyperopt in its basic configurations. Hyperopt
will start by a random hyper parameter combination and will use
$\frac{3}{4}$ as the quantile to obtain $y*$ as seen in the tree parzen
estimation section. For each dataset we performed 30 iterations of
parameter research.

For abnormal point detection, we compared MILOF, HStree, iforestASD,
online ARIMA, and KitNet.

In the algorithm of **MILOF**, the parameters needed are: the maximal
number of clusters **max-nbr-clusters**, the memory limit size **b**,
and the number of nearest neighbor considered **K**. For each dataset we
have taken :

-   $ 5\leq max-nbr-clusters\leq 15$,

-   $ 5\leq K\leq 15$,

-   and we fixed $b=min (500, n) $ with n the size of the dataset.

MILOF was already implemented by the authors, so we used their
implementations available at @milofgithub.

In the **iforestASD** algorithm, the parameters are the initial window
size **init-W**, the window size **WS** inside which we provide score
and monitor concept drift, the number of trees **ntrees**, the maximal
number of features used for training each tree **features**, and the
sampling size $\phi$. Empirically the sampling training size is fixed to
256 as mentioned by tests made by iforest authors. For each dataset we
have taken:

-   $100 \leq init-W = WS \leq \frac{n}{4} $

-   $15 \leq ntrees \leq 100 $

-   $1 \leq features \leq dim$ with **dim** maximal number of features
    of each instance of the dataset

Fortunately we found a library called Pysad which already has an
implementation of iforestASD but without the concept drift
implementation. Pysad proposed some function which permits us to train
in batch and partially, and provide scores. The Pysad iforestASD class
extends the iforest class of another library called Pyod. Pyod provides
a library for anomaly detection in batch data. Through the iforest class
that extends iforestASD, we were able to use our hyperparameters. For
the concept drift implementation, we prefered the non-parametric
proposition of @Togbe2021 using ADWIN, a concept drift detection
algorithm.

In the **HS-tree** algorithm, the parameters are: the initial window
size **init-W**, the window size **WS** inside which the model is train
to test the next window size (so a window size balance from test window
to training window once all its instances has been tested as we
described on the section dedicated to half-space tree) , the number of
trees **ntrees**, and the maximal **max-depth** of each tree. For each
dataset we have taken hyperparameter in the following range:

-   $100 \leq init-W = WS \leq \frac{n}{4} $

-   $15 \leq ntrees \leq 100 $

-   $15 \leq max-depth \leq 25 $

Similarly, as iforestASD Pysad provided us an implementation of Half
space tree.

In the **Online ARIMA** algorithm, the parameters are: the
autoregressive order **p**, and the differential order **d**.

For each dataset we have taken:

-   $0 \leq d\leq 10 $

-   $0 \leq p \leq 200 $

For the implementation of Online ARIMA we referred ourselves to its
implementation in @arimafdgithub. On their implementation we didn’t find
any possibility to add the differencing order, we contacted the author’s
on their GitHub, but we didn’t get any response. Finally we
differentiated the whole stream on our own, using recursively the
**diff()** function provided by the pandas’ library in order to have the
**d** differencing order of the whole stream before providing it to the
algorithm.

In the **Kitnet** algorithm, the parameters are **m** the maximal
dimension that will have each subset of features, and **init-W** the
number of data instances used to learn the feature mapping (it defines
which features should be in each subset of dimension having at most
**m** elements). **m** also defines the maximal dimension of each mini
subset of similar features which would be studied together. For each
dataset we have taken $1 \leq m\leq 30 $. Fortunately, KitNet was
already implemented on the GitHub of the author, and Pysad had already
integrated it. So we directly used this implementation.
