

Importing dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data collection and Data preprocessing

"""

#Loading the dataset to a Pandas DataFrame
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header = None)

sonar_data.head()

#number of rows and column
sonar_data.shape

sonar_data.describe()
#describe gives the statistical measures of the data.

#to find the number of rock and mine examples in the data
sonar_data[60].value_counts()

#More the data , more accurate the model is

#Now group this data into mine and rock
sonar_data.groupby(60).mean()

#Training our model with data and label. but in unsupervised learning we don't need labels.
#seperating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

#We have successfully sperated the data and the labels

"""Training and Test Data

"""

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1 , stratify= Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)
#we have 187 training data and 21 test data.

"""Model Training --> Logistic Regression

"""

model = LogisticRegression()

#training the logistic regression model with training data
model.fit(X_train,Y_train)
#X_train is the training data and Y_train is the training label

"""Model Evaluation

"""

#accuraccy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuraccy on the training data : ', training_data_accuracy)

#accuraccy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on the test data :' , test_data_accuracy)

"""Making a Predictive system

"""

input_data = (0.0100,0.0171,0.0623,0.0205,0.0205,0.0368,0.1098,0.1276,0.0598,0.1264,0.0881,0.1992,0.0184,0.2261,0.1729,0.2131,0.0693,0.2281,0.4060,0.3973,0.2741,0.3690,0.5556,0.4846,0.3140,0.5334,0.5256,0.2520,0.2090,0.3559,0.6260,0.7340,0.6120,0.3497,0.3953,0.3012,0.5408,0.8814,0.9857,0.9167,0.6121,0.5006,0.3210,0.3202,0.4295,0.3654,0.2655,0.1576,0.0681,0.0294,0.0241,0.0121,0.0036,0.0150,0.0085,0.0073,0.0050,0.0044,0.0040,0.0117)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
  print('the object is a Rock')
else:
  print('The object is a mine')

