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
