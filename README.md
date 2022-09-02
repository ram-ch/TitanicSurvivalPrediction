# TitanicSurvivalPrediction
A classification model for predicting the survival of Titanic passengers

## Introduction   
In this repo we will try to build a classification model for the titanic data set available on kaggle   

## Dataset   
The datasets can be downloaded from [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data). In this project I have used open source [DVC](https://dvc.org/) as a data version control tool. The actual data is stored on google drive and only the .dvc version tracking file is saved on the github repository.   
Benifits:   
* This avoids pushing huge data sets and artifacts to the github repository   
* Enables versioning and tracking of the data files without creating multiple copies of the same data with slighlty different names  
* Enables version control for saved models   
* Enables easy sharing of up-to-date data files among teams   

other approache could be using a RDBMS for data storage:   
* Reading and writing data to the DB by multiple teams can slow down the process.    
* Not suitable for huge images, text, audio and video datasets   
* Version control is not possible with the above approach   

## Notebooks   
The notebooks contain the basic EDA  

## Data Preprocessing  


## Experimentation   
I have used [MLflow](https://mlflow.org/) for maintaining a track of different experiments run. Importants information like model name, parameters and evaluation metrics are logged to mlflow. The results can be viewed on a UI by running the below command    
`mlflow ui`    

