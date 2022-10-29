# Project-KaggleCreditCard


# Purpose of this project
- Compare the performance of sampling dataset vs balancing class weights when you train a classification model on an imbalanced dataset
  - Analytical objective: maximize Area under Precision-Recall Curve (AUPRC)
  - sklearn's implementation = Average Precision (AP)
- Identify ways to speed up sklearn on a large dataset with 285,000 rows

# What I learnt from this project 
## AP vs AUROC
- Compared with AP, AUROC tends to be too optimistic when the positive class is rare
<img src="../main/data/image/2022-10-29-14-23-49.png">
<img src="../main/data/image/2022-10-29-14-23-58.png">

- Why is AUROC too optimistic?
  - ROC curve plots recall vs FPR
  - Precision-Recall curve plots recall vs precision
    - essentially substitutes FPR with precision
  - $FPR = \frac{FP}{FP+TN}$
  - $Precision = \frac{TP}{TP+FP}$
  - In a dataset where the positive class is rare and the negative class is abundant, any weak model can capture a large TN, which will inflate the denominator of FPR
- Alternative: custom scoring
  
## Implementational limitation of balancing class weights
  - Cannot use HalvingGridSearchCV or HalvingRandomSearchCV to speed up training for large imbalanced dataset as candidates chosen in the first few epochs to evaluate gradients might not contain the rare class
## Implementational limitation of sampling techniques
  - Cannot apply sample_weight (e.g. "Amount" column) in fitting as sampling technique will shuffle the dataset while sample_weight is based on the original order of the dataset
## Parallelism in sklearn
- n_jobs parameter in sklearn's estimators ([more details](https://scikit-learn.org/stable/computing/parallelism.html))
  - When the underlying implementation uses [joblib](https://joblib.readthedocs.io/en/latest/parallel.html#thread-based-parallelism-vs-process-based-parallelism), the number of workers (threads or processes) that are spawned in parallel can be controlled via the n_jobs parameter
  - <img src="../main/data/image/2022-10-29-15-46-33.png">
- Options for joblib's parallel backend 
  - "loky", "threading": distribute locally
  - "spark": distribute across spark clusters

----------------------------------


# Data Understanding
> script: **source/[Kaggle_1_load_data.ipynb](https://github.com/WillKWL/Project-KaggleCreditCard/blob/main/source/Kaggle_1_load_data.ipynb)**
- [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download): anonymized transactions made by credit cards in September 2013 by European cardholders
- The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions
- For confidentiality issues, the original features V1, V2, ... V28 are not provided, and only the results of a PCA transformation are provided
- Time and amount for each transaction are also provided for cost-sensitive learning

# Data Preparation
Since the dataset has been preprocessed with PCA, we don't have to do a lot of preprocessing except feature scaling and can focus on comparing the performance of our models. We will use the following pipelines for each approach.
## Sampling using imblearn's ADASYN
> script: **source/[Kaggle_2A_ML_workflow_ADASYN_average_precision.ipynb](https://github.com/WillKWL/Project-KaggleCreditCard/blob/main/source/Kaggle_2A_ML_workflow_ADASYN_average_precision.ipynb)**
<img src="../main/data/image/2022-10-29-12-31-56.png">

## Alternative: Balancing the class weights
> script: **source/[Kaggle_2B_ML_workflow_class_balance_average_precision.ipynb](https://github.com/WillKWL/Project-KaggleCreditCard/blob/main/source/Kaggle_2B_ML_workflow_class_balance_average_precision.ipynb)**
<img src="../main/data/image/2022-10-29-12-33-27.png">

# Modeling
## Shortlist a few models to tune
We first limit our scope to classification models implemented in sklearn with the following properties:
- n_jobs parameter for parallelization given the size of the dataset
- predict_proba() method for applying ensemble methods later on

Based on the below test run on standard hyperparameters, we can further tune the hyperparameters of extra trees classifier and XGBoost classifier to improve the performance of the models.
<img src="../main/data/image/2022-10-29-12-43-14.png">

## Tune hyperparameters of the shortlisted models using RandomizedSearchCV

<img src="../main/data/image/2022-10-29-12-56-22.png">
<img src="../main/data/image/2022-10-29-14-23-13.png">

## Apply ensemble learning to improve the performance of the tuned models
- Methods tested: Adaboost, Bagging, Stacking, Voting
- Final model: Adaboost with Extra Trees classifier as the base estimator
- mean 10-fold cross-validation AP: 0.8381 (Adaboost Extra Trees) vs 0.8298 (Extra Trees)

# Evaluation on test set
> script: **source/[Kaggle_3_test_set_evaluation.ipynb](https://github.com/WillKWL/Project-KaggleCreditCard/blob/main/source/Kaggle_3_test_set_evaluation.ipynb)**
- Final model = Adaboost Extra Trees Classifier
- <img src="../main/data/image/2022-10-29-14-29-58.png">

## AP on test set vs CV result
- AP on test set = 0.8757
- The red dot represents the AP on the test set, while the blue dots represents the AP from 10-fold cross-validation. The model's performance on test set is good and within the range of CV numbers.
<img src="../main/data/image/2022-10-29-14-31-01.png">

- AUROC on test set = 0.9750 (too optimistic)
<img src="../main/data/image/2022-10-29-14-33-33.png">

## Lift and gain chart
Lift in 1st decile is more than 9x and we can capture more than 90% of frauds in the first 10% of the test set.
<img src="../main/data/image/2022-10-29-14-34-38.png">

## Precision-Recall Tradeoff
<img src="../main/data/image/2022-10-29-14-47-10.png">
Based on the tradeoff, we can set a cutoff to achieve, for example 85% recall if we prioritize recall over precision as we care about catching more frauds and false alarms are acceptable. The confusion matrix is as below:
<img src="../main/data/image/2022-10-29-14-48-58.png"> <img src="../main/data/image/2022-10-29-14-48-43.png">

