#we have installed scikitlearn via terminal
#scikit learn provides an array of built in metrics for both classification and regression

import pandas as pd
from sklearn.model_selection import train_test_split
#the above line is used to divide the dataset to training and testing data set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#we have imported to know the accuracy and comparision of the data on how well it is working

df=pd.read_csv('titanic.csv')
#the above line is used to read the dataset that we are using for the model development


#now we need to drop the irrelevant data or columns that  is useless for us in the system or model development
df=df.drop(columns=['Passengerid','Name','Ticket','Cabin'])

#now how you gonna deal with the missing data in the age and embarked in the dataset
df['Age'].fillna(df['Age'].median(),inplace=True)
#inplace is used to make the changes directly on the dataframe

#now lets clean the data that is embarked columns
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

#now lets change the data or encode data with 0 or 1 as per the male and female
df['Sex']=df['Sex'].map({'male':0, 'female':1})
# the above does encode the values in the data frame

df=pd.get_dummies(df, columns=["Embarked"], drop_first=True)
# this line converts the Embarked column into numerical data that formats for machine learning
#pd.dummies are used for the encoding of the data in the column in embarked

print("Prepared data for the Ml is:\n",df.head())

