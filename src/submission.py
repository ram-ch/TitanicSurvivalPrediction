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

from sklearn import ensemble
from sklearn import model_selection

import model_dispatcher
import argparse
import xgboost as xgb

train_df=pd.read_csv("data/train_le.csv")
test_df=pd.read_csv("data/test_le.csv")
test_data=pd.read_csv("data/test.csv")

X=train_df.iloc[:,:-1]
y=train_df.iloc[:,-1]
X_test=test_df

# print(X.shape,y.shape, X_test.shape)
# print(X.columns)
# print(X_test.columns)

# params={'max_depth': 5, 'n_estimators': 300}
params={'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}

clf = ensemble.RandomForestClassifier(n_jobs=-1,random_state=42,verbose=1).fit(X, y)
preds = clf.predict(X_test)

submission=pd.DataFrame()
submission['PassengerId']=test_data.iloc[:,0]
submission['Survived']=preds
print(submission.head())
submission.to_csv("submissions/submission.csv",index=False)
