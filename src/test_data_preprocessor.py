import warnings
warnings.filterwarnings('ignore')
import os
import config
import pandas as pd
import numpy as np
from pickle import dump,load
from sklearn import preprocessing
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
            train_mean_age=29.69911764705882
            df['Age']=df['Age'].fillna(value=train_mean_age)    
        elif col=='Embarked':
            df['Embarked']=df['Embarked'].fillna(value='unknown')    
        elif col=='Fare':
            train_mean_fare=32.204207968574636
            df['Fare']=df['Fare'].fillna(value=train_mean_fare)
        elif col=='Sex':
            df['Sex']=df['Sex'].fillna(value='unknown')
    return df

def ohe_encoder(col,df):     
    transformed_df=pd.DataFrame()
    ohe=preprocessing.OneHotEncoder()
    # load the obj to pkl
    with open(f"src/preprocessor_pkl/ohe_{col}.pkl", "rb") as input_file:
        ohe = load(input_file) 
    transformed = ohe.transform(df[[col]])
    transformed_df[ohe.categories_[0]] = transformed.toarray()    
    for col in transformed_df:
        transformed_df[col]=transformed_df[col].astype(int)
    return transformed_df

def label_encoder(col,df):
    # load the obj to pkl
    with open(f"src/preprocessor_pkl/le_{col}.pkl", "rb") as input_file:
        le = load(input_file) 
    # transform the test     
    df[col] = le.transform(df[[col]])
    df[col]=df[col].astype(int)
    return df


def scaler(col,df):
    scaler=preprocessing.StandardScaler()
    with open(f"src/preprocessor_pkl/sclr_{col}.pkl", "rb") as input_file:
        scaler = load(input_file) 
    df[col]=scaler.transform(df[[col]])
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


def run():
    """
    Method performs the steps to load data,preprocessing,kfold split,save    
    """
    # 1.Load data
    test_data=pd.read_csv("data/test.csv")
    test_data=test_data.drop(['PassengerId','Ticket','Cabin'],axis=1)
       
    # 2. check for nulls  & impute - Pclass=unknown,Sex=unknown,Age=mean of train, Embarked=unknown
    null_cols=null_value_checker(test_data)
    test_data=null_imputer(test_data,null_cols)
    null_cols=null_value_checker(test_data)
    if len(null_cols)==0:
        print("After imputation: No Null")
    
    # Recreate name feature
    test_data=name_feature(test_data)
    test_data=test_data.drop(['Name'],axis=1)


    # Create alone or not
    test_data=family_feature(test_data)
    test_data=test_data.drop(['SibSp','Parch','FamilySize'],axis=1)   

    # create fare bands
    test_data=fare_feature(test_data)

    # create age bands
    test_data=age_feature(test_data)    

    # adding prefix to column values
    for col in ['Pclass','Fare','Embarked','Title']:
        test_data[col] = f'{col}_' + test_data[col].astype(str)     

    # # OHE
    # for col in ['Pclass','Sex','Fare','Embarked','Title']:   
    #     transformed=ohe_encoder(col,test_data)
    #     test_data=pd.concat([test_data,transformed], axis=1)
    #     test_data.drop(col,axis=1,inplace=True)

    # LE
    for col in ['Pclass','Sex','Fare','Embarked','Title']:   
        test_data=label_encoder(col,test_data)

    # new feature
    test_data['Age_Class'] = test_data.Age * test_data.Pclass

    # new feature
    test_data['Fare_Embarked'] = test_data.Fare * test_data.Embarked
        
    # 8. save file
    test_data.to_csv('data/test_le.csv',index=False)
    print(test_data.head())
    # return

if __name__=='__main__':
    run()
    # TODO: try pyspark for data loading and preprocessing