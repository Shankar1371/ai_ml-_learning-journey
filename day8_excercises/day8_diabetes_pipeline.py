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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#this import  is used to standardize a data

#now we are diving the data for features and target
X=df.drop(columns=["Outcome"])
y=df["Outcome"]

#now  we have to split the dataset into test and train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=42)


#scale features
scaler =StandardScaler()
#this is used for the scaler object is now ready to learn the parameters from  my data that i have provided

X_train = scaler.fit_transform(X_train)
#the above line is used to standardization formula to every data point in the x
X_test = scaler.transform(X_test)
#this line does the transformation on your dataset and as it has just the transform we have just the test data that will not have any leakage

print("Training data shape:",X_train.shape)
#printing the training data shape