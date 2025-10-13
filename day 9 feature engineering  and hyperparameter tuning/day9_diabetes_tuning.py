#lets load the data and prepare the data for the model training

import pandas as pd
from sklearn.model_selection import train_test_split
#the above import is used to split the data into train and test split
from sklearn.preprocessing import StandardScaler
#the use of the above line is to implement standardization or z score normalization


df= pd.read_csv('diabetes.csv')
#this loads the data

#print the basic info
print("Shape:",df.shape)
print(df.head())

#now we are splitting the features into X and y
X = df.drop(columns=['Outcome'])
y = df['Outcome'] #all the y is only the outcome column and remaining are for the X

#Scale
scaler=StandardScaler()
X_scaled =scaler.fit_transform(X)
#the above line is used to learn from the training data.
#this is mostly used on training data as that helps in both the learning the scaling parameters and also applying the scaling in one step


#now we arre splitting the dataset into training and test split
X_train,X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state =42)
#here we have split the data by using the scaler data \