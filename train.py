import dvc.api
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model.selection import train_test_split
from sklearn.linear.model import ElasticNet
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

import dvc.api
path='/data/AdSmartABdata.csv'
repo='C:/Users/zefa-n/Documents/ab-test-mlops'
version='v1'
data_url=dvc.api.get_url(path=path,
                         repo=repo,
                         rev=version)

mlfow.set_experiment('demo')  


if __name__ == "__main__":
    warnings.filterwarningd("ignore")
    np.random.seed(40)

    data=pd.read_csv(data_url,sep=",")