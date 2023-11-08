#%%Importing LIBS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
from sklearn import linear_model
from sklearn.metrics import r2_score 

#%% READING THE FILE OF FUEL CONSUMPTION AGAIN

df= pd.read_csv("./FuelConsumptionCo2.csv")

df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

df.head(20)

# %% PLOTTING SECTION FOR ENGINE SIZE VERSUS CO2 EMISSIONS

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# %% PLOTTING SECTION FOR ENGINE SIZE VERSUS CO2 EMISSIONS

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='RED')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()


#%% SPLITTING DATA

# OLD WAY
# np.random.seed(0)
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]

# Using sklearn lib to save time
from sklearn.model_selection import train_test_split
np.random.seed(42)
train, test = train_test_split(cdf, test_size=0.2)

print("train: ",train)
print("test: ",test)


train_x
train_y
#separating y axis from x axis, that means splitting between dependent variable vs independent

# %% Evaluation of error section, using testing dataset, about 20%



#%% EXERCICE 2


#%% EXERCISE 3


#%% EXERCISE 4


#%% SCALING DATA

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler() #Instance of the scalling method

data_x_scaled = scaler.fit_transform(train_x) ## 2 in 1

#or 

scaler.fit(train_x)

train_x_scaled = scaler.transform(train_x)

scaler.data_min_ #attribute given by the scaler instance
scaler.data_max_ #attribute given by the scaler instance

#%% Testing data

#using regr
regr = linear_model.LinearRegression()

regr.fit(train_x_scaled, train_y)
test_x_scaled = scaler.transform(test_x)
test_y_ = regr.predict(test_x_scaled)



#%% CROSS VALIDATION SECTION

# For a robust validation

from sklearn.model_selection import cross_val_score, cross_val_predict

scores_r2 = cross_val_score(regr, train_x, train_y, scoring="r2",cv=5)

# OR

predicted = cross_val_predict(regr, train_x, train_y, cv=5)

scores_r2 = r2_score(train_y, predicted)