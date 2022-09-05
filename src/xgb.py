import optuna
from mlflow import log_metric, log_param
import pandas as pd

from sklearn import ensemble
from sklearn import model_selection
from mlflow import log_metric, log_param
import xgboost as xgb


def run(X,y):       
    parameters = {
            "eta":[0.01,0.015,0.025,0.05,0.1],
            "gamma":[0.05-0.1,0.3,0.5,0.7,0.9,1.0] ,
            "max_depth": [3,5,7,9,12,15,17,25],
            "min_child_weight": [1,3,5,7],
            "subsample": [0.6,0.7,0.8,0.9,1.0],
            "colsample_bytree": [0.6,0.7,0.8,0.9,1.0],
            "lambda": [0.01-0.1,1.0],
            "alpha": [0,0.1,0.5,1.0],    
        }
    xgb_clf = xgb.XGBClassifier(n_jobs=-1,random_state=42)
    clf = model_selection.GridSearchCV(xgb_clf, parameters,scoring='accuracy',refit=True, cv=5, verbose=1).fit(X, y)
    accuracy=clf.cv_results_.best_score_
    best_params=clf.best_params_
    return best_params, accuracy, clf


    


if __name__=='__main__':
    # data
    df=pd.read_csv('data/train_le.csv')
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    best_params, accuracy,clf=run(X,y)
    print(f"Accuracy: {accuracy}")
    
    log_param("Model","XGB")
    log_param("Best Params",best_params)
    log_metric("Accuracy",accuracy)