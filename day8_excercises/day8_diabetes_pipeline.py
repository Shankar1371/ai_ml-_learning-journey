#now we are learning how to load the dataset and explore the dataset

import pandas as pd

#loading the dataset
df=pd.read_csv('diabetes.csv')

#exploring the dataset
print("Shape:",df.shape)
print("Columns:",df.columns)
print("\n First five rows",df.head())
print("\nMissing values:\n",df.isnull().sum())
#here we get the sum of missing values in the dataframe
print("\nSummary:\n",df.describe())
#this is used to describe the data frame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#this import  is used to standardize a data

#now we are diving the data for features and target
X=df.drop(columns=["Outcome"])
y=df["Outcome"]

#now  we have to split the dataset into test and train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=42)


#scale features
scaler =StandardScaler()
#this is used for the scaler object is now ready to learn the parameters from  my data that i have provided

X_train = scaler.fit_transform(X_train)
#the above line is used to standardization formula to every data point in the x
X_test = scaler.transform(X_test)
#this line does the transformation on your dataset and as it has just the transform we have just the test data that will not have any leakage

print("Training data shape:",X_train.shape)
#printing the training data shape

#now trainind multiple models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#the above line is used to bring a specific machine  learning algorithm the support vector classifier
#the support vector classifier(SVC) is used to find an optimal hyperplane that maximizes the margin between two classes of data, effectively separating hem to classify the new data points

#first logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#now lets use the decision tree
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train,y_train)

#Random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)


#Support Vector Machine
#SVM is a powerful and a versatile supervised machine learning as the algorithm is used for both for classification and regression


#Support vector classifier
svm =SVC(probability=True, random_state=42)
#here the probability is true this enables the SVC to calcualte the class probabilties for prediction

svm.fit(X_train,y_train)


#evaluvate the models

#that can be achieved by accuracy score and classification_report

from sklearn.metrics import accuracy_score, classification_report

models ={
    "Logistic Regression": log_reg,
    "Decision Tree": dtree,
    "Random Forest": rf,
    "SVM": svm,
}

for name, model in models.items():
    #the above line iterates from each models that are defiened above
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    #the above prints the name3 of the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n",classification_report(y_test,y_pred))

#lets do the ROC curve
#that is the Reciever Operating Characteristic(ROC) curve is used to evaluvate the performance

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#auc is nothing but area under the curve

plt.figure(figsize=(7,6))

for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:,1]
    #the probabilty of getting the y
    fpr,tpr,_ = roc_curve(y_test,y_probs)#this gets the true positive and true negative
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f'{name} (AUC={roc_auc:.2f})')
    #this roc_auc gets the value of AUC that is formated to two decimal places

plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC curve comparision")
plt.legend()
plt.show()
