import pandas as pd

df = pd.read_csv("iris.csv")

#renaming the column

df.rename(columns={"sepal_length":"SepalLength"}, inplace=True)
#inplace=true  to maodify the original dataframe but not create a new dataframe

#filter rows(we are just fjiltering out where the species is setosa
setosa_df =df[df["species"]=="setosa"]

#sorting by special length
sorted_df=df.sort_values(by="SepalLength",ascending=False)

#drop a column

df_no_sepal= df.drop(columns=["sepal_width"])

print("filteredZ(setosa):\n",setosa_df.head())
print("\nSorted by SepalLength:\n",sorted_df.head())
print("\n Dropped a column\n",df_no_sepal.head())
