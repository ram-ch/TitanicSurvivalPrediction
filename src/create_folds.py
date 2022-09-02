import warnings
warnings.filterwarnings('ignore')
import os
import config
import pandas as pd
import numpy as np

from sklearn import model_selection

def create_folds(data):
    data['kfold']=-1
    data=data.sample(frac=1).reset_index(drop=True)
    
    kf=model_selection.KFold(n_splits=5)

    for fold,(trn_,val_) in enumerate(kf.split(X=data)):
        data.loc[val_, 'kfold']=fold
    
    for i in range(5):
        print(f"Fold: {i} | {len(data['kfold']==i)}")
    return data


if __name__=='__main__':
    # 1.Load data
    train_data=pd.read_csv('data/train_le.csv')
    # 2.create folds
    train_data_folded=create_folds(train_data)
    # 3. save the train data with folds
    train_data_folded.to_csv('data/train_le_processed.csv')