import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor as regressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Read data
raw_data = pd.read_csv('train_32.csv',header=0)


# Dividing training set and test set
m = np.size(raw_data,0)
n = np.size(raw_data,1)

X = raw_data.values[:,1:n-1]
Y = raw_data.values[:,n-1]

train_x, test_x, train_y, test_y = train_test_split(X, Y,train_size=0.7,shuffle='false')

train_column = n-2
test_row = m//5
train_row = m-test_row
test_column = train_column

mean_train = np.mean(train_x,0)
max_train = np.max(train_x,0)
min_train = np.min(train_x,0)

# Feature scaling only with training set
train_x = (train_x-mean_train)/(max_train-min_train)
test_x = (test_x-mean_train)/(max_train-min_train)

# Build the structure
print ('MLPregressor:')


nn3 = regressor(hidden_layer_sizes=[64,32,16], max_iter=1000, solver='adam', verbose =True, shuffle=False,activation='relu',
                learning_rate_init=0.002,early_stopping=False,learning_rate='invscaling',alpha = 10000)


# Fit in the training set
nn3.fit(train_x,train_y)

# Prediction
predict_y3 = nn3.predict(test_x)

# Error
e_nn3 = predict_y3 - test_y




# Plot ROC curve
plt.figure()
plt.plot(range(len(predict_y3)),predict_y3,'b',label="NN")
plt.plot(range(len(test_y)),test_y,'r',label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("the index of houses")
plt.ylabel('the price of houses')
plt.show()