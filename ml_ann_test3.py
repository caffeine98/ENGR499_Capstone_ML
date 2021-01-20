import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

data = pd.read_excel('myData5.xlsx', na_values=0)
#data = pd.read_csv('myData1.csv')
print(data.head())
print(data.columns[4])

water = data.columns[2]
cement = data.columns[4]
fine_aggr = data.columns[16]
course_aggr = data.columns[17]
comp_strength_28days = data.columns[28]
flyash = data.columns[5]
silica = data.columns[23]
mult_data = data[[comp_strength_28days, water, cement, fine_aggr, course_aggr]]
#mult_data = mult_data.fillna(0)
mult_data = mult_data.dropna()
print(mult_data)

xVars = mult_data.drop('28 days', axis = 1)
print(xVars)
yVars = mult_data[['28 days']]
print(yVars)

X_train, X_test, y_train, y_test = train_test_split(xVars, yVars, test_size=0.1)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#
model = Sequential()
model.add(Dense(4, input_shape=(4,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1,))
model.compile(Adam(lr=0.0005), 'mean_squared_error')

# Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

# Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
history = model.fit(X_train, y_train, epochs = 5000, validation_split = 0.1,shuffle = True, verbose = 0, callbacks = [earlystopper])
#history = model.fit(X_train, y_train, epochs = 5000, validation_split = 0.2,shuffle = True, verbose = 0)

# Plots 'history'
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='training loss val')


# Runs model with its current weights on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

inputprediction = [[128, 250, 1033.6, 723]]
inputprediction = preprocessing.scale(inputprediction)
prediction = model.predict(inputprediction)
print(prediction)

plt.show()