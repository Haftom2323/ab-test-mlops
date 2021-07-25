#import necessary packages
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import scipy
from scipy import stats

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
    def X_Boost(df):
        data_x = df.loc[:, df.columns != 'response']
        data_y = df['response']
        X_train, X_test, y_train, y_test\
            = train_test_split(data_x, data_y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test\
            = train_test_split(X_test, y_test, test_size=0.10, random_state=1)
        xb = XGBClassifier()
        xb.fit(X_train, y_train)
        print(f"XGBoost accuracy  score = {xb.score(X_test, y_test)}%")
    
        xb_results = cross_val_score(xb, X_train, y_train, cv=5)
        print(f"XGBoost K=5 mean score accuracy = {round(xb_results.mean() * 100,2)} %")
    
        # Log mlflow attributes for mlflow UI
        mlflow.log_metric("accuracy", xb.score(X_test, y_test))
        #mlflow.log_metric("K=5 mean score accuracy", round(xb_results.mean()) )
        mlflow.sklearn.log_model(xb, "model")

    X_Boost(response)