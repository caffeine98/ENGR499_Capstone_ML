import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("myData.csv", na_values=[0])

#print(df.head())
# 'Fine Aggregate (kg/m3)', 'Coarse Aggregate (kg)'
# 'Water (kg/m3)', 'Cement (kg/m3)', 'Fine Aggregate (kg/m3)', 'Coarse Aggregate (kg)', '28 days'
water = data.columns[2]
cement = data.columns[3]
fine_aggr = data.columns[10]
course_aggr = data.columns[11]
comp_strength_28days = data.columns[20]
flyash = data.columns[4]
silica = data.columns[15]

#print(data.columns[3])

data1 = data[[comp_strength_28days, cement]]
data2 = data[[comp_strength_28days, water]]
data3 = data[[comp_strength_28days, fine_aggr]]
data4 = data[[comp_strength_28days, course_aggr]]
data5 = data[[comp_strength_28days, flyash]]
data6 = data[[comp_strength_28days, silica]]

#print(data1)
#data2 = data[[fine_aggr, comp_strength_28days]]
#print(data2)

mult_data = data[[comp_strength_28days, water, cement, fine_aggr, course_aggr, flyash, silica]]

print(mult_data)
print(mult_data.shape)

#plt.figure()
#figure, axes = plt.subplots(2,2)
#data1 = fig.add_subplot(221)
#data2 = fig.add_subplot(222)
#data3 = fig.add_subplot(223)
#data4 = fig.add_subplot(224)

data1.plot(kind = 'scatter', x=comp_strength_28days, y=cement)
data2.plot(kind = 'scatter', x=comp_strength_28days, y=water)
data3.plot(kind = 'scatter', x=comp_strength_28days, y=fine_aggr)
data4.plot(kind = 'scatter', x=comp_strength_28days, y=course_aggr)
data5.plot(kind = 'scatter', x=comp_strength_28days, y=flyash)
data6.plot(kind = 'scatter', x=comp_strength_28days, y=silica)


#mult_data.plot(kind = 'scatter', x=comp_strength_28days, y=cement)

plt.show()

#reg = linear_model.LinearRegression()
#reg.fit(mult_data[[water, cement, fine_aggr, course_aggr]], comp_strength_28days)