import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#load the dataset on to the program
df = sns.load_dataset("iris")

#now we need to plot a heatmap (correlation heatmap)
corr =df.corr(numeric_only=True)
#df.corr() this function calculates the coorelation between  all the columns in a dataframe
#numeric_only=true is a parameter that ensures the calculation is only performed on columns with numerical data
sns.heatmap(corr, annot=True, cmap="coolwarm")
#this above function is to get the heatmap
#annot=true adds the corelation values directly onto each cell of the heat map

plt.title("Correlation Heatmap")
plt.show()

#pairplot

sns.pairplot(df, hue="species")
plt.show()
