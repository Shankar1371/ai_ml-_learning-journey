#This is a mini project that is and gives a basic EDA on iris dataset
#EDA is a Exploratory Data Analysis is the process of examining and summarizing the main characteristics of a dataset often through
#visual and statistical methods.

import pandas as pd

import seaborn as sns
#what is seaborn
#Seaborn is a opensorce python data visualization library that is built on top of matplotlib
#its primary purpose is to simplfy the creation of attractive and informative statistical graphics
#this kind of representation makes the data exploration and presentation more efficient
#it has specifically designed for creating a statistical plots such as scater plots line plots and bar plots
#histograms etc

import matplotlib.pyplot as plt
#matplotlin.pyplot is a module with in the Matplotlib library in python that provides a MATLAB like interface for creating a static and interactive and animated visualizations
#this can be used for creating a plottypes and customizing plo  elements such as tiles and axis labels etc

df=pd.read_csv("iris.csv")
#the above line reads the dataset that is provided

print(df.describe())
#this will give the summary of the dataset iris that have been used in the file

print("\n Species count:\n",df["species"].value_counts())
#this will give the different species and their count that are available in the dataset

sns.pairplot(df,hue="species")
#seaborn.pairplot is a function that uses a grid for visualize pairwise  relationships with in a dataset
#so this pairplot will create a grid of subplots for all numerical values

#hue="species" is an optional but a powerful argument that adds a semantic mapping to the plot by colouring the data points
plt.show()