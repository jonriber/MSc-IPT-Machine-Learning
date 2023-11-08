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
df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

df.head(100)

# %%
