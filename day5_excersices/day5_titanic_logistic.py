import form
#today we are gonna learn our first machine learning model
#so we will prepare a data set , train a logistic regression model
#and finally evaluvate its performance
#what scikit_learn
#it is an open source python library used for machine learning. it provides as a comprehensive set of tools for various machine learning
#and that is primarlily focused on traditional machine learning algorithm rather than deep learning


# goal for today is simple ML pipeline_> preprocess-> train -> test ->evaluvate
import pandas as pd

from sklearn.model_selection import train_test_split
#the above line train_test_split is used to divide the dataset into training and testing subsets, which is a crucial step in machine learning to evaluvate the unseen data and the model performance of it

from sklearn.linear_model import LogisticRegression
#logistic regression is a class where the module uses regression algoristhm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# imports various functions for evaluvating  the performance of the trained model
#like accuracy score ,confusion mqatrix
#and classifcation_report

df=pd.read_csv("titanic.csv")
#loading the csv dataset

df=df.drop(columns=["PassengerId","Name","Ticket","Cabin"])
#dropping all the irrelevant columns from the dataframe that we are not using in the dataframe

#handling the missing values dataset
df["Age"].fillna(df["Age"].median(),inplace=True)
#here the age is filled which is null with the median of the age  and the inplace argument ensureues the change is directly made on the dataframe


df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

df["Sex"] = df["Sex"].map({"male":0, "female":1})
#the above line is used for encoding as machine learning algorithms work with numbers but not text
df= pd.get_dummies(df,columns=["Embarked"], drop_first=True)
#as coding

print("prepared Data:\n", df.head())
#the above steps is used to clean the dataset for the ml

#splitting the data
#the features(X) and target(Y)

X = df.drop(columns=["Survived"])
#the remaining columns are stored in x
y=df["survived"]
#survived is in y


#train and test split( 80% tain and 20% test)
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2, random_state=42)


#traing the mode with logistic regression model

#Create a model

model = LogisticRegression(max_iter=500)

#now train the model
model.fit(X_train,y_train)

print("Model trainig completed")

