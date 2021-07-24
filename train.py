import dvc.api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix,plot_confusion_matrix

from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, auc, \roc_auc_score

import mlflow
import mlflow.sklearn

import os
import warnings
import sys
import pathlib
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("./data").resolve()

path = DATA_PATH.joinpath("platformData.csv")
repo = 'D:/ab-test-mlops/model.py'
version = 'v2'

data_url = dvc.api.get_url(path=path,
                           repo=repo,
                           rev=version)

mlflow.set_experiment("Logistic Regression")

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    response = pd.read_csv('platformData.csv')
    response['awareness'] = response.yes
    response.drop(['yes', 'no', 'auction_id'], axis=1, inplace=True)

    lb_make = LabelEncoder()
    # encode the categorical variables 
    response['experiment']=lb_make.fit_transform(response['experiment'])
    response['date']=lb_make.fit_transform(response['date'])
    response['device_make'] = lb_make.fit_transform(response['device_make'])
    #response['browser'] = lb_make.fit_transform(response['browser'])
    if 'browser' in response.columns:
        response['browser'] = lb_make.fit_transform(response['browser'])
    else:
        response=response
    # split the data into train, validation and test
    train, validate, test = np.split(response.sample(frac=1, random_state=42),
                                     [int(.7 * len(response)), int(.9 * len(response))])
    # separate the response variable from the dataset
    y_train, y_validation, y_test = (train['awareness'], validate['awareness'], test['awareness'])
    x_train, x_validation, x_test = (train.drop(['awareness'], axis=1), validate.drop(['awareness'], axis=1),
                                     test.drop(['awareness'], axis=1))

    max_depth=None
    max_features='auto'
    max_samples=None
    n_estimators=100

    # Run ElasticNet
    clf_rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=max_depth,
                                              max_features=max_features,
                                              max_leaf_nodes=None,
                                              max_samples=max_samples,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=n_estimators, n_jobs=None,
                                              oob_score=False,
                                              random_state=None, verbose=0,
                                              warm_start=False)
    log_result = clf_rf.fit(x_train, y_train)
    predictions_log = clf_rf.predict(x_validation)
    (accuracy, precision, recall) = eval_metrics(y_validation, predictions_log)
        # Report training set score
    train_score = clf_rf.score(x_train, y_train) * 100
    # Report test set score
    test_score = clf_rf.score(x_validation, y_validation) * 100

   
    # Print Random Forest model metrics
    print("  Accuracy: %s" % accuracy)
    print("  Precision: %s" % precision)
    print("  Recall: %s" % recall)

    # Log mlflow attributes for mlflow UI
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(clf_rf, "model")

    