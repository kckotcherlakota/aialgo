#Importing Libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#Importing DataSet
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleanig the data
import re                  #Simple library to clean data
import nltk                #lib with pacakges to remove unwanted words      
nltk.download('stopwords')  #downloading stopwords(Unwanted words) from nltk            
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  #keeps root of a word
corpus = [] #pack of words
for i in range(0, 1000):                    #for all dataset
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #keeping letters from A_Z
    review = review.lower()                 #changing into lower
    review = review.split()                 #into list and seperating words  
    ps = PorterStemmer()                   
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  #each words checked and remove unwanted
    review = ' '.join(review)
    corpus.append(review)

#Creating bag of words - unique words 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   #spasing max_featurs to reduce 0's in matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(55+91)/200 * 100  #Accuracy (correct pred)/(whole test)