#loading the data and preparing that
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#the above line we have taken or imported the train and test split from sklearn

df=pd.read_csv("titanic.csv")
#loading the dataset

#droping the irrelevant columns
df=df.drop(columns=["PassengerId","Name","Ticket","Cabin"])

#now we are handling the missing data or values
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

#now we need to encode the data for the models
#and that can be achieved by encoding
df["Sex"]=df["Sex"].map({"male":1, "female":0})
df=pd.get_dummies(df,columns=["Embarked"], drop_first=True)

#features and target
#now we split the data into training and testing dataset
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#now we are doing a cross validation
#cross vcalidation is a technique that is used to evalvate on how well a model will generalize to new aned unseen data.
#it is also the most robust method than a simple train and test split because it uses all the abailable data for both training and valiidation of method

from sklearn.model_selection import cross_val_score
#the cross_val score is used to evaluvate the performance of the machine learning model using cross validation.
#it provides a simple and convenient way to get a robust estimate of a models performace on unseen data without manually coding the cross -validation loop


from sklearn.ensemble import RandomForestClassifier
#the use of the Random forest classifier is an ensemble learning method used for classification tasks
#it works by building a large number of individual decision trees and combining their predictions to get a simple and more accurate and stable result


model = RandomForestClassifier(random_state=42,n_estimators=100)
#random state is used to ensure the reproducibilty as it relies on random number generator
#n_estimators is used to set the number of decision trees that have to be used in a model

cv_scores= cross_val_score(model,X_train,y_train,cv=5)

#here we are using 5 -fold cross validation and that is set by cv=5
#and the model is used by the top line
#this line evaluvates or cross evaluvates the model 5 times by using the equal folds of data that will be helpful for cross validation


print("Cross validation score:",cv_scores)
#these shoes the score from cross validation on 5 folds
print("Mean of cross validation score:",cv_scores.mean())
#this line shows the median of all the scores



#cross validation checks the model performance on multiple splits and reducing bias from one lucky and unlucky split


#ROC curve and AUC
#ROC(Recevier operating Characteristics)
#the roc curve shows It is a graph that illustrates the performance of the binary classification model at various classification threshoilds.

#AUC(This is area under the curve of ROC)

from sklearn.metrics import roc_curve,auc
#we have imported the required library to display the roc and auc curve

#and before that we need to import the matplotlib.pyplot  as we need to show the plot that is required

model.fit(X_train,y_train)

#predicting the probabilities
y_probs= model.predict_proba(X_test)[:,1]
#it gets the probility of predicting the colums by the x_test that wijll be in time of just one second [:,1]


#to gets roc curve based on the y_probability and y_test of the test split
#and that will use FPR and TPR (that is false positive rate and true positive rate

fpr,tpr,thresholds = roc_curve(y_test,y_probs)
#thresholds is an array of probability  thresholds used to calculate the FPR and TPR

roc_auc=auc(fpr,tpr)

#auc summarizes an AUC of 1.0 represnets the perfect classifier
#and AUC of 0.5 is no better than  the random guessing


#we will learn how to plot the ROC curve using the library matplot lib that is imported as plt
#ROC curve

plt.plot(fpr,tpr,label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],linestyle='--',color='gray')
#the arguments [0,1] and [0,1] these do have specificates the base line and refrence line of the plot that we are using in the ROC curve
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
plt.legend()
#legend on the plot helps us with different colors patterns or symbols that helps use to represenr the data series in a graph
plt.show()
