
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Convolution1D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
#from sklearn.decomposition import PCA
from random import randint

import sys

def normalize(X):
    # Find the min and max values for each column
    x_min = 1e100
    x_max = -1e100
 
    for i in range (X.shape[0]):
        for j in range (X.shape[1]):
            if(X[i,j]>x_max):
                x_max = X[i,j]
            if(X[i,j]<x_min):    
                x_min = X[i,j]

    print('max ',x_max)
    print('min ',x_min)
#    x_min = -1000.0
 #   x_max = 1000.0
    for i in range (X.shape[0]):
        for j in range (X.shape[1]):
            if(X[i,j]>x_max):
                print ('need to select new max value')
                print('X[i,j] ',X[i,j])
                input ('wait ')
            if(X[i,j]<x_min):    
                print ('need to select new min value')
                print('X[i,j] ',X[i,j])
                input ('wait ')
    # Normalize

    for i in range (X.shape[0]):
        for j in range (X.shape[1]):
            X[i,j] = (X[i,j]-x_min)/(x_max-x_min)
    return X

            
def main(data_traning, labels_training, data_testing, labels_test, submission_path,maxEpoch):

    # loading data
    data_train = pd.read_csv(data_traning) 
    data_test = pd.read_csv(data_testing) 

    data_train_labels = pd.read_csv(labels_training) 
    data_test_labels = pd.read_csv(labels_test) 

    train_data = data_train.values[:, 0:]
    train_labels = data_train_labels.values[:, 0]

    test_data = data_test.values[:,0:] #[:, 1:]
    test_labels = data_test_labels.values[:,0] # [:,1]
    
    convNN = True
   
 #   train_data = normalize(train_data)
 #   test_data = normalize(test_data)
    

    # create NN
    model = Sequential()

    if(convNN):
        dataset_Size= train_data.shape[0]
        num_inputs= train_data.shape[1]
        
        train_data = np.expand_dims(train_data, axis=2)
        test_data = np.expand_dims(test_data, axis=2)

        model.add(Convolution1D(32, 10,activation='relu', input_shape=(num_inputs,1)))
        model.add(MaxPooling1D(pool_length=10, stride=None, border_mode='valid'))
   #     model.add(Convolution1D(32, 10, border_mode='same'))
  #      model.add(MaxPooling1D(pool_length=35, stride=None, border_mode='valid'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='linear'))
   #     model.add(Flatten())

    else:    
        model.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
        model.add(Dense(5, activation='tanh'))
     #   model.add(Dense(3, activation='tanh'))
        model.add(Dense(1, activation='linear'))
    
    print('train_data.shape[1] ',train_data.shape[1])
    print('train_data.shape[0] ',train_data.shape[0])
    print('train_data ',train_data)
    print('model out ',model.summary())
    print('model out ',np.shape(model.predict(train_data)))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc', 'mae'])

    # train NN
    model.fit(train_data, train_labels, epochs=maxEpoch, verbose=2)



    # Get training Accuracy
    result = model.predict(train_data)
    print('len(result) ',len(result))
    count=0
    predictions = np.zeros((len(result),), dtype=np.int)

    for i in range (len(result)):
        ans=0
        if(result[i]<0.5):
            ans = 0
        else:
            ans = 1
        predictions[i] = ans
        if (ans==train_labels[i]):
            count=count+1

    TrainAccuracy = (float(count)/float(len(result)))*100.0

    print('Train Accuracy = ',TrainAccuracy)
    result_dataframe = pd.DataFrame(result, columns=['NN Output'])
    result_dataframe['Weekend_Day_Predict'] = predictions
    result_dataframe['Weekend_Day_Actual'] = train_labels
    result_dataframe.index.name = 'id'
    outfile = submission_path+'_train.csv'
    result_dataframe.to_csv(outfile)

    # Get test Accuracy
    
    result_test = model.predict(test_data)

    count=0
    predictions_test = np.zeros((len(result_test),), dtype=np.int)
    for i in range (len(result_test)):
        ans=0
        if(result_test[i]<0.5):
            ans = 0
        else:
            ans = 1
        predictions_test[i] = ans
        if (ans==test_labels[i]):
            count=count+1

    TestAccuracy = (float(count)/float(len(result_test)))*100.0
    
    print('Test Accuracy = ',TestAccuracy)
    result_test_dataframe = pd.DataFrame(result_test, columns=['NN Output'])
    result_test_dataframe['Weekend_Day_Predict'] = predictions_test
    result_test_dataframe['Weekend_Day_Actual'] = test_labels
    result_test_dataframe.index.name = 'id'
    outfile = submission_path+'_test.csv'
    result_test_dataframe.to_csv(outfile)
    return (TrainAccuracy, TestAccuracy)
    

if __name__ == "__main__":
    folds = 2
    sumTrainAcc = 0
    sumTestAcc = 0
    maxEpoch = 20 #500
    for k in range(folds):
        
        
        data_training = 'SystemDemand_Jan19_features.csv' 
        labels_traning = 'SystemDemand_Jan19_labels.csv' 
        data_testing = 'SystemDemand_Feb19_features.csv' 
        labels_testing = 'SystemDemand_Feb19_labels.csv' 
        
        submission_path = 'C:/Users/FILEPATH/fold_'+str(k)+'_DNN_predictions' 

        TrainAcc, TestAcc = main(data_training, labels_traning, data_testing, labels_testing, submission_path,maxEpoch)
        sumTrainAcc = sumTrainAcc+TrainAcc
        sumTestAcc = sumTestAcc+TestAcc

    avgTrainAcc = float(sumTrainAcc)/float(folds)
    avgTestAcc = float(sumTestAcc)/float(folds)
    print('avgTrainAcc ',avgTrainAcc)
    print('avgTestAcc ',avgTestAcc)
    
















