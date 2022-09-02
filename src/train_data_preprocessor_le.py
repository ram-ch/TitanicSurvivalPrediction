import warnings
warnings.filterwarnings('ignore')
import os
import config
import pandas as pd
import numpy as np
from pickle import dump,load
from sklearn import preprocessing

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

# def ohe_encoder(col,df):     
#     transformed_df=pd.DataFrame()
#     ohe=preprocessing.OneHotEncoder()
#     # fit to train
#     ohe.fit(df[[col]])            
#     transformed = ohe.transform(df[[col]])
#     # save the obj to pkl
#     dump(ohe, open(f'{config.PREPROCESSOR_PKL}/ohe_{col}.pkl', 'wb'))    
#     transformed_df[ohe.categories_[0]] = transformed.toarray()    
#     for col in transformed_df:
#         transformed_df[col]=transformed_df[col].astype(int)
#     return transformed_df

def scaler(col,df):
    scaler=preprocessing.StandardScaler()
    scaler.fit(df[[col]])
    df[col]=scaler.transform(df[[col]])
    # save the obj to pkl
    dump(scaler, open(f'{config.PREPROCESSOR_PKL}/sclr_{col}.pkl', 'wb'))    
    return df


def run():
    """
    Method performs the steps to load data,preprocessing,kfold split,save    
    """
    # 1.Load data
    train_data=pd.read_csv(config.RAW_TRAIN_DATA)
       
    # 2. check for nulls  & impute - Pclass=unknown,Sex=unknown,Age=mean of train, Embarked=unknown
    null_cols=null_value_checker(train_data)
    train_data=null_imputer(train_data,null_cols)
    null_cols=null_value_checker(train_data)
    if len(null_cols)==0:
        print("After imputation: No Null")
    
    # # 3. adding prefix to column values
    # for col in ['Pclass','SibSp','Parch','Embarked']:
    #     train_data[col] = f'{col}_' + train_data[col].astype(str)
    
    # # 4. OHE
    # for col in ['Pclass','SibSp','Parch','Embarked','Sex']:   
    #     transformed=ohe_encoder(col,train_data)
    #     train_data=pd.concat([train_data,transformed], axis=1)
    
    # # 4. label Encoder
    # for col in ['Pclass','SibSp','Parch','Embarked','Sex']:   
    #     transformed=ohe_encoder(col,train_data)
    #     train_data=pd.concat([train_data,transformed], axis=1)

    from sklearn import preprocessing
    for col in ['Sex','Cabin','Embarked']:
        le = preprocessing.LabelEncoder()
        le.fit(train_data[col])
        train_data[col]=le.transform(train_data[col])
    
    # 5. drop columns - Name, Ticket
    drop_columns=['PassengerId','Ticket','Name','Cabin']
    train_data.drop(drop_columns,axis=1,inplace=True)
    
    # 6. standard scaling 'Age' and 'Fare'
    for col in ['Age','Fare']:
        train_data=scaler(col,train_data)
    
    # 7. rearrage column
    surv=train_data.pop('Survived')
    train_data['Survived']=surv

    # 8. save file
    train_data.to_csv('data/train_le.csv',index=False)
    print(train_data.head())
    return

if __name__=='__main__':
    run()
    # TODO: try pyspark for data loading and preprocessing