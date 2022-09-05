#
# use ./run.sh to run multiple models and get base line scores
#

import warnings
warnings.filterwarnings('ignore')
import os
import config
import argparse
import math
import pandas as pd
import numpy as np
import joblib
from statistics import mean
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import model_dispatcher
from mlflow import log_metric, log_param, log_artifacts, start_run


def run(model,df):
    X=df.drop(["Survived"], axis=1).values
    y=df.Survived.values

    # fit and predict
    clf=model_dispatcher.models[model]
    clf.fit(X,y)
    score = model_selection.cross_val_score(clf, X, y, scoring='accuracy',n_jobs=-1, cv=15,verbose=False)
    accuracy = score.mean()
    print(f"{model}: {accuracy}")
    print("-"*50)
    return accuracy

if __name__=='__main__':
    # try any specific model
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str)
    args=parser.parse_args()
    df=pd.read_csv('data/train_le.csv')
    accuracy=run(args.model,df)
    model=args.model
    # log ml flow metrics
    start_run(run_name='Baseline Models')
    log_param("Model",model)
    log_metric("Accuracy", accuracy)