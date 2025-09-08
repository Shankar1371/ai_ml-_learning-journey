import pandas as pd
#as usual we have imported the pandas
#and today we gonna learn about on how to load a csv file
df =pd.read_csv("iris.csv")

print("shape",df.shape)
print("columns:",df.columns)
print("\n first 5 rows:\n",df.head(5))
print("\n Summary stats:\n",df.describe())
print("\n Missing values:\n",df.isnull().sum())