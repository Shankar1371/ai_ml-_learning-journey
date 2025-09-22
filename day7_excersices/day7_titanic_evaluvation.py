#loading the data and preparing that
import pandas as pd
from sklearn.model_selection import train_test_split
#the above line we have taken or imported the train and test split from sklearn

df=pd.read_csv("titanic.csv")
#loading the dataset

#droping the irrelevant columns
df=df.drop(columns=["passengerId","Name","Ticket","Cabin"])

#now we are handling the missing data or values
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

#now we need to encode the data for the models
#and that can be achieved by encoding
df["Sex"]=df["Sex"].map({"male":1, "female":0})
df=pd.get_dummies(df,columns=["Embarked"], drop_first=True)

#features and target
#now we split the data into training and testing dataset
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)