#now we are learning how to load the dataset and explore the dataset

import pandas as pd

#loading the dataset
df=pd.read_csv('diabetes.csv')

#exploring the dataset
print("Shape:",df.shape)
print("Columns:",df.columns)
print("\n First five rows",df.head())
print("\nMissing values:\n",df.isnull().sum())
#here we get the sum of missing values in the dataframe
print("\nSummary:\n",df.describe())
#this is used to describe the data frame

