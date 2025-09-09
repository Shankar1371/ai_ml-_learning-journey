import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset("iris")
#this is a seaborn dataset that has been loaded using the abobe line to load the dataset

#histogram
sns.histplot(df["sepal_length"], kde=True)
#this will plot a histogram that is a seaborn function
#and this df['sepal_length'] determines the data that needs to be plotted
#kde stands for kernel density estimate
#and setting that to true tells seaborn to compute the draw a smooth density curve
plt.title("Histogram for sepal length")
#this sets the title for the graph or the plot that has been created
plt.show()

#scatterplot
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
plt.title("Scatter plot for sepal length")
plt.show()

#boxplot
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Box plot for sepal length")
plt.show()
