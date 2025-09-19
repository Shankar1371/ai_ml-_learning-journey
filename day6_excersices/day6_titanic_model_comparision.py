#we have installed scikitlearn via terminal
#scikit learn provides an array of built in metrics for both classification and regression

import pandas as pd
from PIL.TiffImagePlugin import Y_RESOLUTION
from sklearn.model_selection import train_test_split
#the above line is used to divide the dataset to training and testing data set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from day5_excersices.day5_titanic_logistic import X_train

#we have imported to know the accuracy and comparision of the data on how well it is working

df=pd.read_csv('titanic.csv')
#the above line is used to read the dataset that we are using for the model development


#now we need to drop the irrelevant data or columns that  is useless for us in the system or model development
df=df.drop(columns=['PassengerId','Name','Ticket','Cabin'])

#now how you gonna deal with the missing data in the age and embarked in the dataset
df["Age"].fillna(df["Age"].median(),inplace=True)
#inplace is used to make the changes directly on the dataframe

#now lets clean the data that is embarked columns
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

#now lets change the data or encode data with 0 or 1 as per the male and female
df['Sex']=df['Sex'].map({'male':0, 'female':1})
# the above does encode the values in the data frame

df=pd.get_dummies(df, columns=["Embarked"], drop_first=True)
# this line converts the Embarked column into numerical data that formats for machine learning
#pd.dummies are used for the encoding of the data in the column in embarked

print("Prepared data for the Ml is:\n",df.head())



#now we need to split the dataset into training and testing dataset

#this has Features(X) and Target(y)

X=df.drop(columns=["Survived"])
#so x has all the dataset that  drops the surivived as that would be the target
Y=df["Survived"]
#and whu will be the result that we have got

#and then after training with what is the result that is  y we take the test split and then we compare the data that we got with the actual dataset
#that will give how good is the model that we have go

#now we need tpo split the dataset that is 80 and 20 percent

X_train,X_test,Y_Train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#this divides the data in the most random way possible

print("the train shape:",X_train.shape)
print("the test shape:",X_test.shape)

#now lets train multiple models that we use this system and that would be logistic regression, Random Forest  and decision tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#this line used for the importing the decision tree library
from sklearn.ensemble import RandomForestClassifier
#the above line is used for the importing the library for random forest library for developing the model



#now training a model with logistic regression
log_reg=LogisticRegression(max_iter=500)
#maxiter is a parameter that is used for te maxiumum number if iterations for the solver to convergence .
#and this helps to prevent the model from getting stuck in an infinte loop
log_reg.fit(X_train,Y_Train)
#the above line trains the logistic regression
#log_reg.fit(X,Y) as it is a supervised learning that has been used
Y_pred_log=log_reg.predict(X_test)
#the prediction  is done by giving only gthe data that is there for xtest
#and then comparing the data by testing x with the model
#later we compare the output that we got to the actual y_test that tells us how good the system


#lets learn about the decision tree
#what is a decision tree
#decsion tree is a supervised machine learning algorithm that ises a tree-like model for decisions and their possible consequences.
#it is used for both classification and regression
dtree=DecisionTreeClassifier(random_state=42, max_depth=5)
#here the above line is used to create and train the decsion tree model that has the random forest and maxdepths set
#random_state=42 is the parameter that is set for number generator that is random in the dataset that we are using
#max_depths is the hyperparameter that controls the complexity of the decision tree and that limits the maximum depths of the tree
dtree.fit(X_train,Y_Train)
#this gives the data for the model that it needed to be trained
y_pred_tree=dtree.predict(X_test)


#lets go and learn about the random forest
#what is random forest in ai
#it is powerful supervised machine learning algorithm that uses ensemble learning method.
#the random forest is an extension to decision tree and producess the accurate prediction

#uses or advantages of random forest in the ai
# High accuracy , reduces overfitting and handles various data types
#the disadvantages are computationally expensive and have more complexity to the solution

rf=RandomForestClassifier(random_state=42,n_estimators=100,max_depth=5)
#n_estimators is used to specify the number of individual decision tress to be created in the forest
rf.fit(X_train,Y_Train)
y_pred_rf=rf.predict(X_test)


