import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


data = pd.read_csv("myData2.csv")

water = data.columns[2]
cement = data.columns[3]
fine_aggr = data.columns[10]
course_aggr = data.columns[11]
comp_strength_28days = data.columns[20]
flyash = data.columns[4]
silica = data.columns[15]

mult_data = data[[water, cement, fine_aggr, course_aggr, flyash, silica, comp_strength_28days]]
X = mult_data.iloc[0:6]
y = mult_data.iloc[6]#.values
print(X.shape)
print(y.shape).values
sc = MinMaxScaler()
X = sc.fit_transform(X)
y= y.reshape(-1,1)
y = sc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=13, input_dim=13))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)

results=regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''



X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)
'''
data1 = data[[comp_strength_28days, cement]]
data2 = data[[comp_strength_28days, water]]
data3 = data[[comp_strength_28days, fine_aggr]]
data4 = data[[comp_strength_28days, course_aggr]]
data5 = data[[comp_strength_28days, flyash]]
data6 = data[[comp_strength_28days, silica]]

print(mult_data)
'''