# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:37:57 2018

@author: Ashish Saha
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('spam.csv' ,encoding='latin-1')
X=dataset.iloc[:,1].values
y=dataset.iloc[:,0].values
X=pd.DataFrame(X)
y=pd.DataFrame(y)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y[0] = labelencoder.fit_transform(y[0])


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', X[0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6200)
Z= cv.fit_transform(corpus).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Z_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(Z_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#check accuracy of model
confidence = classifier.score(Z_test, y_test)# accuracy is 86 %ge



# Fitting DecisionTree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier1.fit(Z_train,y_train)
# Predicting the Test set results
y_pred_1=classifier1.predict(Z_test)
confidence_1= classifier.score(Z_test, y_test)# accuracy is also 86% ge




# Fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(criterion='entropy', random_state=0)
classifier2.fit(Z_train,y_train)
# Predicting the Test set results
y_pred_3=classifier2.predict(Z_test)
#check accuracy of model
confidence_3= classifier.score(Z_test, y_test)






