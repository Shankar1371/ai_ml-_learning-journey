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

#now we are learning about the feature engineering
#lets add a few derived features that is simple and may  reveal the hidden relationships

import numpy as np

df["BMI_Age"] = df["BMI"] * df["Age"]
#this above line creates a new colum from the existing columns
df["Glucose_BMI"] = df["Glucose"] / (df["BMI"] +1)

print(df[["BMI", "Age", "BMI_Age","Glucose_BMI"]].head())


#hyperparameter tuning with grid search cv
#we will tune the randomforest since its flexible and strong
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#this gridsearch is a crucial tool used for hyperparameter tuning in machine learning . it is primary use is to systematically searc for the optimal combination of hyperparameter
#for the given model to maximize the performance


rf=RandomForestClassifier(random_state=42)


param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

#what is the use of the above line that is used to define the search space for the gridserach
#thus giving th eblueprint that tells the gridsearch  exactly which hyperparameter combinations to test

#n_estimators is the number of the treess
#max_depth is used to give the depth of the each tree
#min-sample split and this is used to gives the samples that are required to be present at a node
#min_samples_leaf gives the threshold  that contains the leaf


grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1,verbose=1)
#estimators is used to tune is likely holds an instance of a randomforest
#param grid  is the hyperparameters that we have given above
#cv is called cross validation folds. the model will be  trained and evaluvated 5 times for every combination of hyperparameters
#scoring is given as accuracy as it is the evaluavtion metric used to judge which combination that is best
#n_jobs is used for the number of cores that are the used
#verbose mostly contrils the amount of text output that will be eseen during the fitting pprocess


grid_search.fit(X_train,y_train)

print("Best parameters:", grid_search.best_params_)
#this gives the best parameters for the system that is beeen caluclated and grid_search.best_params
print("best CV Accuracy:", grid_search.best_score_)


#the parameters and the output show the best Hyperparameter

#now evaluvating for the best model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

best_model =grid_search.best_estimator_
#this is used to find or retrieve the best performing model found in a grid search process in machine learning
#.best_estimator is a attribute of the grid search object and after the grid search completes its evaluvation of all hyperparmeters combinations
#and that line  is used for the best_estimator and this code with all the best hyperparameters combination

y_pred = best_model.predict(X_test)

print("\n Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix",confusion_matrix(y_test, y_pred))
print("\n Classification Report",classification_report(y_test, y_pred))

