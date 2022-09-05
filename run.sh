#!/bin/sh
echo "Preprocessing.."
python src/train_data_preprocessor.py
python src/test_data_preprocessor.py
echo "Training..."

for model in knn log_reg svm dt rf gb nn xgb;
do
    python src/baseline.py --model $model    
done
echo "MlFlow Dashboard"
# mlflow ui