#in day 4 we are gonna learn about putting everything together
#like pandas and visualization into Full EDA (Exploratory Data Analysis) workflow

#data overview
import pandas as pd

df=pd.read_csv("titanic.csv")

#here we have loaded the data using the pandas library


#basic overiew of the data
print("shape:",df.shape)
print("\n columns: \n",df.columns)
print("\n First five rows: \n",df.head(5))
print("\n Data Types: \n",df.dtypes)
print("\n Missing values: \n ",df.isnull().sum())


#learning about cleaning data in the titanic

#firstly  we will drop all the irrelevant columns
df  =   df.drop(columns=["PassengerID","Name","Ticket","Cabin"])


#now here we have to handle the missing data
df["Age"].fillna(df["Age"].median(),inplace=True)
#in the above line the in age indecies we are filling the null values with some of the values from the embarked and this also fills the missing values
#.mode is used to take the most frequently used . inplace is used to modify it directly
df["embarked"].fillna(df["Embarked"].mode()[1],inplace=True)

print("\n After cleaning missing values:\n",df.isnull().sum())