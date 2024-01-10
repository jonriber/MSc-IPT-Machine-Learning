
#%% Importing libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%% Reading dataset

df_drug = pd.read_csv("./drug200.csv")

#%% Reading the first 6 lines
df_drug.head()

#%% Checking for null/missing values
print(df_drug.info())
#%% CATEGORICAL VARIABLES
# TO-DO HERE: discuss about ordinal categorical variable and non-ordinal    
df_drug.Drug.value_counts()
df_drug.Sex.value_counts()
df_drug.BP.value_counts()
df_drug.Cholesterol.value_counts()
#%% NUMERICAL VARIABLES
# mean count, std, min, max and others using describe function
df_drug.describe()

skewAge = df_drug.Age.skew(axis = 0, skipna = True)
print('Age skewness: ', skewAge)

skewNatoK = df_drug.Na_to_K.skew(axis = 0, skipna = True)
print('Na to K skewness: ', skewNatoK)

sns.distplot(df_drug['Age'])
sns.distplot(df_drug['Na_to_K'])
# %% DRUG TYPE DISTRIBUTION

sns.set_theme(style="darkgrid")
sns.countplot(y="Drug", data=df_drug, palette="flare")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()

# %%
