from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    # Clean and one hot encode data
    df = data.to_pandas_dataframe().dropna()
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df['Sex']=le.fit_transform(df['Sex'])
    df['ChestPainType']=le.fit_transform(df['ChestPainType'])
    df['RestingECG']=le.fit_transform(df['RestingECG'])
    df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
    df['ST_Slope']=le.fit_transform(df['ST_Slope'])
    x_df=df.iloc[:,:11]
    y_df = df.pop("HeartDisease")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    
    ds =  TabularDatasetFactory.from_delimited_files(['https://raw.githubusercontent.com/sireeshag09/udacity-capstone/main/heart.csv'])
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", np.float(AUC_weighted))

if __name__ == '__main__':
    main()
