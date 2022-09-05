import optuna
from mlflow import log_metric, log_param
import pandas as pd

from sklearn import ensemble
from sklearn import model_selection
from mlflow import log_metric, log_param



def run(X,y):       
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
    parameters = {'n_estimators':[200,250,300],
                  'max_depth':[2,3,5],
                  'min_samples_split':[1,2],
                  'min_samples_leaf':[2,5],
                  'max_features':('sqrt',None)
                  }
    rf = ensemble.RandomForestClassifier(n_jobs=-1,random_state=42,verbose=1)
    # clf = model_selection.GridSearchCV(rf, parameters,scoring='accuracy',refit=True, cv=5).fit(X, y)
    # accuracy=clf.cv_results_.mean_test_score
    # best_params=clf.best_params_
    # return best_params, accuracy, clf

    clf = model_selection.GridSearchCV(rf, parameters,scoring='neg_log_loss',refit=True, cv=5).fit(X, y)
    logloss=clf.best_score_
    best_params=clf.best_params_
    return best_params, logloss, clf
    


if __name__=='__main__':
    # data
    df=pd.read_csv('data/train_le.csv')
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    # best_params, accuracy,clf=run(X,y)
    # print(f"Accuracy: {accuracy}")

    best_params, logloss,clf=run(X,y)
    print(f"Log Loss: {logloss}")

    log_param("Model","RF")
    log_param("Best Params",best_params)
    log_metric("Log loss",logloss)