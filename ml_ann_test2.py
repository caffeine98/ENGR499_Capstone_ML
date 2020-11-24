import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

data = pd.read_excel('myData1.xlsx', na_values=0)
#data = pd.read_csv('myData1.csv')
print(data.head())
print(data.columns[4])

water = data.columns[2]
cement = data.columns[3]
fine_aggr = data.columns[10]
course_aggr = data.columns[11]
comp_strength_28days = data.columns[20]
flyash = data.columns[4]
silica = data.columns[15]

mult_data = data[[comp_strength_28days, water, cement, fine_aggr, course_aggr, flyash, silica]]
mult_data = mult_data.fillna(0)
print(mult_data)

xVars = mult_data.drop('28 days', axis = 1)
#print(xVars)
yVars = mult_data[['28 days']]
#print(yVars)

xTrain, xValid, yTrain, yValid = train_test_split(xVars, yVars ,train_size=0.8, random_state=2)

scaler = MinMaxScaler()
scaler.fit(xTrain)

xTrain = scaler.transform(xTrain)
xValid = scaler.transform(xValid)

print(pd.DataFrame(xTrain).describe())
print(pd.DataFrame(xValid).describe())

#lbfgs for small datasets, adam for large datasets
nn = MLPRegressor(hidden_layer_sizes=(100,100,100,100,), activation='relu', max_iter=1024, solver='adam')
nn.fit(xTrain, yTrain)

mae = metrics.mean_absolute_error(yTrain, nn.predict(xTrain))
mse = metrics.mean_squared_error(yTrain, nn.predict(xTrain))
rsq = metrics.r2_score(yTrain, nn.predict(xTrain))
print('Training Data: ')
print(mae, mse, rsq)


maeV = metrics.mean_absolute_error(yValid, nn.predict(xValid))
mseV = metrics.mean_squared_error(yValid, nn.predict(xValid))
rsqV = metrics.r2_score(yValid, nn.predict(xValid))
print('Testing Data: ')
print(maeV, mseV, rsqV)
#
#predictions = nn.predict(xValid)
#print(predictions)