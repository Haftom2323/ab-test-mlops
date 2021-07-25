#import necessary packages
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from urllib.parse import urlparse
import scipy
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix,plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import scipy
from scipy import stats

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dvc.api

import os
import warnings
import sys
import pathlib
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


path = 'data/browser.csv'
repo = 'C:/Users/zefa-n/Documents/ab-test-mlops/'
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

    response = pd.read_csv(data_url,sep=",")
    
    lb_encoder = LabelEncoder()
    # encode the categorical variables 
    response['experiment']=lb_encoder.fit_transform(response['experiment'])
    response['date']=lb_encoder.fit_transform(response['date'])
    response['device_make'] = lb_encoder.fit_transform(response['device_make'])
    if 'browser' in response.columns:
        response['browser'] = lb_encoder.fit_transform(response['browser'])
    else:
        response=response
    #scale data
    scaler = MinMaxScaler()
    def scale(df):
        scalled = scaler.fit_transform(df)
        scalled_df = pd.DataFrame(data = scalled, columns=df.columns)
        return scalled_df
    response=scale(response)   
    def Logistic_reg(df):
        data_x = df.loc[:, df.columns != 'response']
        data_y = df['response']
        X_train, X_test, y_train, y_test\
            = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test\
            = train_test_split(X_test, y_test, test_size=0.10, random_state=1)
        with mlflow.start_run(): 
            C=1   
            lr = LogisticRegression(C=C, random_state=0)
            lr.fit(X_train, y_train)
            prediction = lr.predict(X_val)
            (accuracy, precision, recall) = eval_metrics(y_val, prediction)

            print("Logistic Regression model (C=%f):" % (C))
            print("  Accuracy: %s" % accuracy)
            print("  Precision: %s" % precision)
            print("  Recall: %s" % recall)
            
            lr_results = cross_val_score(lr, X_train, y_train, cv=5)
            score=round(lr_results.mean() * 100,2)
            print(f"Linear Regression K=5 mean score accuracy = {score} %")
         # Log mlflow attributes for mlflow UI
            # Log mlflow attributes for mlflow UI
            mlflow.log_param("C", C)
            mlflow.log_metric("accuracy", accuracy)
            #print(accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(lr, "model")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
           
            # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="Logistc_regression")
        else:
            mlflow.sklearn.log_model(lr, "model")

    Logistic_reg(response)