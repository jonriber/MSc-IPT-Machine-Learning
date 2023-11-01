#%% IMPORTING PANDAS, NUMPY AND DATASET
import pandas as pd
import numpy as np
Heart_Data = pd.read_csv("heart_attack_prediction_dataset.csv")

#%%SHOW full DATA
Heart_Data.head(10000)
#%% TESTING HOW TO READ PEACES OF THE DATA
Heart_Data.head(5) #FIRST 5 ROWS
Heart_Data.tail(5) #LAST 5 ROWS

#%% USEFULL METHODS TO UNDERSTAND DATASET
Heart_Data.info()
Heart_Data.columns.values
Heart_Data.shape

#%% DATA SUMMARY and getting a column value mean fast
Heart_Data.describe()

Heart_Data.describe()["Age"]["mean"]

#%% GETTING SPECIFIC DATA COLUMNS OF DATASET
##These are my target variables and data of interest for correlation
Heart_physical = Heart_Data["Physical Activity Days Per Week"]
Heart_age = Heart_Data["Age"]
Heart_sex = Heart_Data["Sex"]
Heart_country = Heart_Data["Country"]

#%% 
