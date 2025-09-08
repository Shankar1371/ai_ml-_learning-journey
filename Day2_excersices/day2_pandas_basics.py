import pandas as pd
#here we are importing pandas so what is pandas
#pandas is a open source widely used for data manipulation and analysis
#it provides high [erformance easy to use data  structures and machine learning work flows

data ={
    "Name": ["alice","bob","charlie","david"],
    "Age":[22,30,35,40],
    "city": ["NY","LA","Chicago","Houston"]
}

df = pd.DataFrame(data)
#what is a data frame
#a data frame is the pandas library for python is a two dimensional data structure
#the key characteritsics of data frame
#that are two dimensional structure
#data is organized in rows and columns providing tabular representations

print("dataframe:\n",df)
print("\n column names:\n",df.columns)
print("\nFirst two rows:\n",df.head(2))
#head function is used to get top  are inspect the beggining of a dataframe
print("\n describe data:\n ",df.describe())
