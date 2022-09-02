import warnings
warnings.filterwarnings('ignore')
import os
import config
import math
import pandas as pd
import numpy as np
import joblib
from statistics import mean

from sklearn import linear_model
from sklearn import metrics
from mlflow import log_metric, log_param, log_artifacts
from sklearn.tree import DecisionTreeClassifier
import model_dispatcher
import argparse


def run(model,df,fold):
    # train val split
    df_train=df[df.kfold!=fold].reset_index(drop=True)
    df_valid=df[df.kfold==fold].reset_index(drop=True)

    x_train=df_train.drop("Survived", axis=1).values
    y_train=df_train.Survived.values

    x_valid=df_valid.drop("Survived",axis=1).values
    y_valid=df_valid.Survived.values
    # fit and predict
    clf=model_dispatcher.models[model]
    clf.fit(x_train,y_train)
    preds=clf.predict(x_valid)
    # eval
    accuracy=metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model
    joblib.dump(clf,config.MODEL_OUTPUT+f'/{model}.bin')
    return accuracy

if __name__=='__main__':
    # try any specific model
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str)
    args=parser.parse_args()
    

    if args.model=="log_reg" :
        df=pd.read_csv('data/train_ohe_processed.csv')
    else:
        df=pd.read_csv('data/train_le_processed.csv')
    acc=[]
    for i in range(5):
        acc.append(run(args.model,df,fold=i))
    avg_acc=mean(acc)
    print("-"*20)
    print(f"kfold Accuracy: {avg_acc}")
    # log ml flow metrics
    log_param("model",args.model)
    log_metric("Accuracy", avg_acc)