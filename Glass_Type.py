# Load libraries
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import normalize
from scipy import stats

names = ['id','ri','na','mg','al','si','k','ca','ba','fe','type_of_glass']
dataset = pandas.read_csv('glassdata.csv',names=names)

dataset = dataset.drop('id',axis = 1) # Do not use ID column
#peek of 10 records

print(dataset.head(10))

print(dataset.isnull().sum().sort_values(ascending=False).head())

#summary of dataset

print(dataset.describe())

#Dimension of dataset

print(dataset.shape)

#Class of target variable

print(dataset.groupby("type_of_glass").size())

# Split-out validation dataset
dataset.iloc[:,4] = np.log(dataset.iloc[:,4])
print(dataset.head())
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
validation_size = 0.30
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


'''Lets evaluate 6 different algorithms:
    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM).
'''

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "MSG : %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


rfc = RandomForestClassifier(n_estimators = 150, max_depth=10)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
#print("predictions:",X_validation," Y_validation : ",Y_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
