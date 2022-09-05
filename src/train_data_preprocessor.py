import warnings
warnings.filterwarnings('ignore')
import os
import config
import pandas as pd
import numpy as np
from pickle import dump,load
from sklearn import preprocessing
from sklearn import model_selection

def null_value_checker(df):
    null_cols=[]
    col_list=list(df.columns)
    for col in col_list:
        null_count=df[col].isnull().sum()
        if null_count>0:
            print(f"{col} | Null Count: {null_count}")
            null_cols.append(col)
    return null_cols

def null_imputer(df,null_cols):
    # impute missing values
    for col in null_cols:
        if col=='Age':    
            df['Age']=df['Age'].fillna(value=df['Age'].mean())    
        elif col=='Embarked':
            df['Embarked']=df['Embarked'].fillna(value='unknown')    
        elif col=='Fare':
            df['Fare']=df['Fare'].fillna(value=df['Fare'].mean())
        elif col=='Sex':
            df['Sex']=df['Sex'].fillna(value='unknown')
    return df

def ohe_encoder(col,df):     
    transformed_df=pd.DataFrame()
    ohe=preprocessing.OneHotEncoder()
    # fit to train
    ohe.fit(df[[col]])            
    transformed = ohe.transform(df[[col]])
    # save the obj to pkl
    dump(ohe, open(f'{config.PREPROCESSOR_PKL}/ohe_{col}.pkl', 'wb'))    
    transformed_df[ohe.categories_[0]] = transformed.toarray()    
    for col in transformed_df:
        transformed_df[col]=transformed_df[col].astype(int)
    return transformed_df

def label_encoder(col,df):
    le=preprocessing.LabelEncoder()
    # fit to train
    le.fit(df[[col]])            
    df[col] = le.transform(df[[col]])
    # save the obj to pkl
    dump(le, open(f'{config.PREPROCESSOR_PKL}/le_{col}.pkl', 'wb'))    
    df[col]=df[col].astype(int)
    return df


def scaler(col,df):
    scaler=preprocessing.StandardScaler()
    scaler.fit(df[[col]])
    df[col]=scaler.transform(df[[col]])
    # save the obj to pkl
    dump(scaler, open(f'{config.PREPROCESSOR_PKL}/sclr_{col}.pkl', 'wb'))    
    return df

def name_feature(df):
    name=df['Name']
    df['Title']=[i.split('.')[0].split(',')[-1].strip() for i in name]
    for i in range(len(df['Title'])):
        if df['Title'][i]=='Mr':
            df['Title'][i]='Mr'
        elif df['Title'][i]=='Mrs':
            df['Title'][i]='Mrs'
        elif df['Title'][i]=='Miss':
            df['Title'][i]='Miss'
        elif df['Title'][i]=='Master':
            df['Title'][i]='Master'
        else:
            df['Title'][i]='unknown'
    return df

def family_feature(df):
    df['IsAlone']=0
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

def fare_feature(df):
    df['FareBand'] = pd.qcut(df['Fare'], 4)
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    df = df.drop(['FareBand'], axis=1)
    return df

def age_feature(df):
    df['AgeBand'] = pd.cut(df['Age'], 5)
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']
    df = df.drop(['AgeBand'], axis=1)
    df['Age'] = df['Age'].astype(int)
    return df

def create_folds(df):
    df['kfold']=-1
    df=df.sample(frac=1).reset_index(drop=True)    
    kf=model_selection.KFold(n_splits=5)
    for fold,(trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold']=fold    
    for i in range(5):
        print(f"Fold: {i} | {len(df['kfold']==i)}")
    return df


def run():
    """
    Method performs the steps to load data,preprocessing,kfold split,save    
    """
    # Load data
    train_data=pd.read_csv(config.RAW_TRAIN_DATA)
    train_data=train_data.drop(['PassengerId','Ticket','Cabin'],axis=1)
       
    # check for nulls  & impute - Pclass=unknown,Sex=unknown,Age=mean of train, Embarked=unknown
    null_cols=null_value_checker(train_data)
    train_data=null_imputer(train_data,null_cols)
    null_cols=null_value_checker(train_data)
    if len(null_cols)==0:
        print("After imputation: No Null")
    
    # Recreate name feature
    train_data=name_feature(train_data)
    train_data=train_data.drop(['Name'],axis=1)

    # Create alone or not
    train_data=family_feature(train_data)
    train_data=train_data.drop(['SibSp','Parch','FamilySize'],axis=1)    
    
    # create fare bands
    train_data=fare_feature(train_data)

    # create age bands
    train_data=age_feature(train_data)    

    # adding prefix to column values
    for col in ['Pclass','Fare','Embarked','Title']:
        train_data[col] = f'{col}_' + train_data[col].astype(str)    

    
    # OHE
    # for col in ['Pclass','Sex','Fare','Embarked','Title']:   
    #     transformed=ohe_encoder(col,train_data)
    #     train_data=pd.concat([train_data,transformed], axis=1)
    #     train_data.drop(col,axis=1,inplace=True)
    
    # LE
    for col in ['Pclass','Sex','Fare','Embarked','Title']:   
        train_data=label_encoder(col,train_data)
    
    # new feature
    train_data['Age_Class'] = train_data.Age * train_data.Pclass

    # new feature
    train_data['Fare_Embarked'] = train_data.Fare * train_data.Embarked

    # rearrage column
    surv=train_data.pop('Survived')
    train_data['Survived']=surv

    # create folds
    # train_data=create_folds(train_data)

    # save file
    train_data.to_csv('data/train_le.csv',index=False)
    
    return

if __name__=='__main__':
    run()
    # TODO: try pyspark for data loading and preprocessing