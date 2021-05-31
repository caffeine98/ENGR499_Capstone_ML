import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Read Concrete Mix Dataset
data = pd.read_csv('myData10.csv')

#Extracting all required input parameters
#Water, Cement, Fine aggregate, Fineness modulus, Silica, Superplasticizer
water = data.columns[0]
cement = data.columns[1]
fine_aggr = data.columns[2]
finenessmod = data.columns[3]
silica = data.columns[4]
superplasticizer = data.columns[5]

#Extracting all output parameters
#Compressive Strength and Slump
comp_strength_28days = data.columns[6]
slump = data.columns[7]

mult_data = data[[comp_strength_28days, slump, water, cement, fine_aggr, finenessmod, silica, superplasticizer]]
# convert str to int for column '28 days'
mult_data['28 days'] = pd.to_numeric(mult_data['28 days'], errors='coerce')
mult_data = mult_data.dropna(axis=0)

def neural_network_model():
    # Inputs X
    xVars = mult_data.drop(['28 days', 'Slump'], axis = 1).values
    # Outputs Y
    Y = mult_data[['28 days', 'Slump']].values

    #Data is split, 90% for training and 10% for validating
    X_train, X_test, y_train, y_test = train_test_split(xVars, Y, test_size=0.1)

    #Using Keras Functional API for a multiple input multiple output model
    #1 input layer, 5 hidden layers, 1 output layer
    visible = Input(shape=(len(mult_data.columns) - 2,))
    hidden1 = Dense(64, activation='relu')(visible)
    hidden2 = Dense(32, activation='relu')(hidden1)
    hidden3 = Dense(16, activation='relu')(hidden2)
    hidden4 = Dense(16, activation='relu')(hidden3)
    hidden5 = Dense(8, activation='relu')(hidden4)
    output1 = Dense(2, activation='relu')(hidden5)

    model = Model(inputs=visible, outputs=output1)
    #Learning rate = 0.001, Optimizer = Adam
    model.compile(Adam(lr=0.001), 'mean_squared_error')
    print(model.summary())

    # Use an Earlystopper function to stop the script when the model stops improving significantly
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')

    # Executes model over 1000 iterations and implements the earlystopper function.
    history = model.fit(X_train, y_train, epochs = 1000, validation_split = 0.1, callbacks = [earlystopper])

    # Plots the mean square error losses for training and validating sets
    history_dict=history.history
    loss_values = history_dict['loss']
    val_loss_values=history_dict['val_loss']
    plt.plot(loss_values,'b',label='Training loss')
    plt.plot(val_loss_values,'r',label='Validation loss')
    plt.legend()

    # Runs model with its current weights on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculates and prints R2 scores of training and testing data
    training_r2 = r2_score(y_train, y_train_pred)
    training_r2 = round(training_r2,2)
    testing_r2 = r2_score(y_test, y_test_pred)
    testing_r2 = round(testing_r2,2)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
    
    #Save model to drive
    model.save('trained_model.h5')

    plt.show()
    plt.savefig('accuracy.png', dpi = 400)

    return training_r2, testing_r2

