import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#this project combines the EDA(exploratory data analysis) with real world dataset

#loading the dataset

df=pd.read_csv("titanic.csv")

#find out the survival count of the titanic dataset
sns.countplot(x="Survived",data=df)
#this data=df argument specifies the data source will be pandas dataframe
plt.title("Survived count in titanic dataset")
plt.show()


#survival by gender
sns.countplot(x="Survived",hue="Sex",data=df)
#hue argument helps to divide the color to show each combination of two categories
plt.title("Survived count by gender in titanic dataset")
plt.show()


#age distribution in the titanic dataset
sns.histplot(df["Age"].dropna(),bins=20,kde=True)
#KDE(kernel density estimate gives the curve to the age column in the data frame  and the argument is set to true
plt.title("Age distribution in titanic dataset")
plt.show()

#correlation heat map
#lets do the app again
corr = df.corr(numeric_only=True)
#here we have given the numberic_only argument as now we can see the the smoothing of the data that is corelated which has the value as numbers
#now we need to develop the heat map for the corr that we have to develop and we need to use the lib seaborn
sns.heatmap(corr, annot=True, cmap="coolwarm")

#annot is used to use correlated data 'directly on the heatplot  and that can be achieved by annot
plt.title("Correlation of titanic dataset with heatmap")
plt.show()