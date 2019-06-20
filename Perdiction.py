# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:31:52 2018

@author: Ananda Mohon Ghosh
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor as regressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import History 
from keras import losses as keras_loss
import keras.optimizers as keras_optmz
import keras.activations as keras_activf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


#dataframeEncoded = pd.read_csv("Encoded.csv", header=1)
#datasetEncoded = dataframeEncoded.values

#dataframeTrain = pd.read_csv("train.csv", header=1)
#datasetTrain = dataframeTrain.values

#dataframeTest = pd.read_csv("test.csv", header=1)
#datasetTest = dataframeTest.values

#dataframeEncoded.fillna(dataframeEncoded.mean())
#dataframeEncoded.to_csv('out.csv', header = 1)

#raw_data.isnull().values.any()
#nan_rows = df[df.isnull().T.any().T]

# Dividing training set and test set
raw_data = pd.read_csv('TrainData_84_features.csv',header=1)
m = np.size(raw_data,0)
n = np.size(raw_data,1)
X = raw_data.values[:,0:n-1]
Y = raw_data.values[:,n-1]
train_column = n-1


#mean_train = np.mean(X,0)
#max_train = np.max(X,0)
#min_train = np.min(X,0)

# Feature scaling with training set
#train_x = (X-mean_train)/(max_train-min_train)

raw_data_test = pd.read_csv('TestData_84_features.csv',header=1)
o = np.size(raw_data,1)
X_test = raw_data_test.values[:,0:o-1]
Y_test = raw_data_test.values[:,o-1]

p = np.concatenate((X, X_test), axis=0)
q = np.concatenate((Y, Y_test), axis=0)

mean_train = np.mean(p,0)
max_train = np.max(p,0)
min_train = np.min(p,0)
r = (p-mean_train)/(max_train-min_train)


plt.figure(figsize=(15,8))
plt.plot(range(len(Y)),Y,'*',label="Housing price")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()


train_x, test_x, train_y, test_y = train_test_split(r, q,train_size=0.8, shuffle='true')
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.8, shuffle='true')


#Linear Regression
LReg = LinearRegression().fit(train_x, train_y)
pred_LReg = LReg.predict(test_x)
e_LReg = np.absolute(test_y - pred_LReg)
plt.figure(figsize=(15,8))
plt.plot(range(len(pred_LReg)),pred_LReg,'b',label="Linear Regression")
plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()

LR_error = mean_squared_error(test_y, pred_LReg, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("LR:",LR_error)
np.savetxt("e_LReg_hist.csv", e_LReg, delimiter=",")


##Logistic Regression
#logReg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
#logReg.fit(train_x, train_y)
#logReg_pred = logReg.predict(test_x)
#e_reg = test_y - logReg_pred
#plt.figure(figsize=(15,8))
#plt.plot(range(len(logReg_pred)),logReg_pred,'b',label="Logistic Regression")
#plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
#plt.legend(loc="upper right")
#plt.xlabel("index no")
#plt.ylabel('price of houses')
#plt.show()



#KNN
knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(train_x, train_y)
knn_pred = knn.predict(test_x)
e_knn = np.absolute(test_y - knn_pred)
#plt.figure(figsize=(15,8))
plt.plot(range(len(knn_pred)),knn_pred,'b',label="KNN")
plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()

LR_error = mean_squared_error(test_y, knn_pred, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("KNN:",LR_error)
np.savetxt("e_knn_hist.csv", e_knn, delimiter=",")


# Gradient Boosted Regression Trees (GBRT)
#X ,y = make_friedman1 (random_state=150, noise=2.0)
gbrt_reg = GradientBoostingRegressor(n_estimators=200,max_depth=15, learning_rate=0.01,  random_state=0, loss='ls')
gbrt_reg.fit(train_x, train_y)
GBRT_pred = gbrt_reg.predict(test_x)
e_GBRT = np.absolute(test_y - GBRT_pred)
#plt.figure(figsize=(15,8))
plt.plot(range(len(GBRT_pred)),GBRT_pred,'b',label="GBRT")
plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()
GBRT_error = mean_squared_error(test_y, GBRT_pred, sample_weight=None)
GBRT_error = np.sqrt(GBRT_error)
print ("GBRT:",GBRT_error)
np.savetxt("e_GBRT_200_15_hist.csv", e_GBRT, delimiter=",")


#Random Forest
fandomForest = RandomForestRegressor(n_estimators=227,max_depth=35, random_state=0)
fandomForest.fit(train_x, train_y)
fandomForest_pred = fandomForest.predict(test_x)

e_RF = np.absolute(test_y - fandomForest_pred)
#plt.figure(figsize=(15,8))
plt.plot(range(len(fandomForest_pred)),fandomForest_pred,'b',label="Random Forest")
plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()
LR_error = mean_squared_error(test_y, fandomForest_pred, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("Random Forest:",LR_error)
np.savetxt("e_RF_227_35_hist.csv", e_RF, delimiter=",")

# SVR
svr_reg = SVR(kernel='rbf', C=100000, epsilon=0.2)
svr_reg.fit(train_x, train_y)
SVR_pred = svr_reg.predict(test_x)
#plt.figure(figsize=(15,8))
e_SVR  = np.absolute(test_y - SVR_pred)
plt.plot(range(len(SVR_pred)),SVR_pred,'b',label="SVR Prediction")
plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("index no")
plt.ylabel('price of houses')
plt.show()
SVR_error = mean_squared_error(test_y, SVR_pred, sample_weight=None)
SVR_error = np.sqrt(SVR_error)
print ("SVR:",SVR_error)
np.savetxt("e_SVR_hist.csv", e_SVR, delimiter=",")



# Build the structure
print ('MLPregressor:')
nn3 = regressor(hidden_layer_sizes=[128,64,16],max_iter=1000, solver='adam', verbose =True, shuffle=False,activation='relu', learning_rate_init=0.001,early_stopping=True,learning_rate='invscaling',alpha = 10000)
nn3.fit(train_x,train_y)

predict_y3 = nn3.predict(test_x)
e_nn3  = test_y - predict_y3
plt.figure(figsize=(15,8))
plt.plot(range(len(predict_y3)),predict_y3,'b',label="NN Prediction")
plt.plot(range(len(test_y)),test_y,'r',label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("the index of houses")
plt.ylabel('the price of houses')
plt.show()




##define base model
def base_model():
     input_img = Input(shape=(83,))
     model = Dense(64, activation='relu')(input_img)
     model = Dense(32, activation='relu')(model)
     model = Dense(16, activation='relu')(model)
     model = Dense(1, activation='relu')(model)
     model = Model(input_img, model)
     model.compile(loss=keras_loss.mse, optimizer = 'adam', metrics=['accuracy']) #, metrics=['accuracy']
     return model

seed = 7
np.random.seed(seed)

scale = StandardScaler()
#train_x = scale.fit_transform(train_x)
#train_x = scale.fit_transform(train_x)
train_history = History()   
#clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5,verbose=0, callbacks=[train_history],  shuffle=True)
dnn = base_model()
dnn.fit(train_x,train_y,epochs=100, callbacks=[train_history], shuffle=True, validation_data=[val_x, val_y])
pred_dnn = dnn.predict(test_x)

e_nn3  = test_y - pred_dnn
## line below throws an error
#clf.score(test_y, res)
plt.figure(figsize=(15,8))
plt.plot(np.arange(0, 100), train_history.history["loss"], label="train loss")
plt.plot(np.arange(0, 100), train_history.history["val_loss"], label="val loss")
plt.title("Neural network Loss")
plt.xlabel("Epoch/Iteration")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

plt.figure(figsize=(15,8))
plt.plot(range(len(pred_dnn)),pred_dnn,'b',label="NN Prediction")
plt.plot(range(len(test_y)),test_y,'r',label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("the index of houses")
plt.ylabel('the price of houses')
plt.show()


plt.figure(figsize=(15,8))
plt.plot(range(len(pred_LReg)),pred_LReg,'+',label="Linear Regression")
plt.plot(range(len(fandomForest_pred)),fandomForest_pred,'g',label="Random Forest")
plt.plot(range(len(knn_pred)),knn_pred,'v',label="KNN")
plt.plot(range(len(SVR_pred)),SVR_pred,'m',label="SVR Prediction")
plt.plot(range(len(GBRT_pred)),GBRT_pred,'y',label="GBRT")
plt.plot(range(len(pred_dnn)),pred_dnn,'b',label="NN")
plt.plot(range(len(test_y)),test_y,'r',label="TEST SET")
plt.legend(loc="upper right")
plt.xlabel("the index of houses")
plt.ylabel('the price of houses')
plt.show()




# Evaluating by RMES
print ("######################################")
print ("Evaluating by RMSE:")
nn_error = mean_squared_error(test_y, pred_dnn, sample_weight=None)
nn_error = np.sqrt(nn_error)
print ("NN:", nn_error)

LR_error = mean_squared_error(test_y, pred_LReg, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("LR:",LR_error)

LR_error = mean_squared_error(test_y, fandomForest_pred, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("Random Forest:",LR_error)

LR_error = mean_squared_error(test_y, knn_pred, sample_weight=None)
LR_error = np.sqrt(LR_error)
print ("KNN:",LR_error)

SVR_error = mean_squared_error(test_y, SVR_pred, sample_weight=None)
SVR_error = np.sqrt(SVR_error)
print ("SVR:",SVR_error)

GBRT_error = mean_squared_error(test_y, GBRT_pred, sample_weight=None)
GBRT_error = np.sqrt(GBRT_error)
print ("GBRT:",GBRT_error)
print ("######################################")


np.savetxt('LR_Pred.csv', pred_LReg, fmt='%.4f', delimiter=',', header="LR_Pred")





# Gradient Boosted Regression Trees (GBRT)
X ,y = make_friedman1 (random_state=10, noise=2.0)
progressArray = []
depth = [2, 5, 10, 15, 25,35]
tree = [50, 100, 150, 200, 250, 30]
for ac in range (0,6):
    ppt = [0,0,0,0,0,0]
    for ab in range(0,6):    
        gbrt_reg = GradientBoostingRegressor(n_estimators=tree[ac], learning_rate=0.01, max_depth=depth[ab], random_state=0, loss='ls')
        gbrt_reg.fit(train_x, train_y)
        GBRT_pred = gbrt_reg.predict(test_x)
        e_GBRT = test_y - GBRT_pred
        plt.plot(range(len(GBRT_pred)),GBRT_pred,'b',label="GBRT")
        plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
        plt.legend(loc="upper right")
        plt.xlabel("index no")
        plt.ylabel('price of houses')
        plt.show()
        GBRT_error = mean_squared_error(test_y, GBRT_pred, sample_weight=None)
        GBRT_error = np.sqrt(GBRT_error)
        
        print ("######################################")
        print ("GBRT:",GBRT_error)
        ppt[ab] =GBRT_error 
        plt.show()
        print ("######################################")
    progressArray.append(ppt)





# Random Forest (Rf)
X ,y = make_friedman1 (random_state=10, noise=2.0)
progressArray = []
depth = [2, 5, 10, 15, 25,35]
tree = [50, 100, 150, 200, 250, 300]
for ac in range (0,6):
    ppt = [0,0,0,0,0,0]
    for ab in range(0,6):    
        fandomForest = RandomForestRegressor(max_depth=37, random_state=0, n_estimators=tree[ac])
        fandomForest.fit(train_x, train_y)
        fandomForest_pred = fandomForest.predict(test_x)
        LR_error = mean_squared_error(test_y, fandomForest_pred, sample_weight=None)
        LR_error = np.sqrt(LR_error)
        
        plt.plot(range(len(GBRT_pred)),GBRT_pred,'b',label="Tree ={0}  Depth = {1} Error = {2}".format(tree[ac],  depth[ab], format(LR_error, '.4f') ))
        plt.plot(range(len(test_y)),test_y,'r', label="TEST SET")
        plt.legend(loc="upper right")
        plt.xlabel("index no")
        plt.ylabel('price of houses')
        
        print ("######################################")
        print ("Random Forest:",LR_error)
        ppt[ab] =LR_error 
        plt.show()
        print ("######################################")
    progressArray.append(ppt)
