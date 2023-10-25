import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#GET THE DATASET and download it from the link
# get_ipython().system('wget --no-check-certificate -O FuelConsumption.csv https://cf-coursesdata.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101ENSkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv')


#use pandas to extract the information

df = pd.read_csv("C:\DEV\MESTRADO\MACHINE_LEARNING\FICHAS\FP03\FuelConsumptionCo2.csv")

#data size and dimension
np.shape(df)
np.shape(df)[0]
np.shape(df)[1]

#take a look at the dataset

df.head()
df.info()

df.describe()
df.describe()["ENGINESIZE"]
df[["ENGINESIZE","CYLINDERS"]].describe()


#filtering only cars of one model, AUDI for example

mask = df["MAKE"] == "AUDI"

AUDI_CARS = df[mask] #GET ONLY TRUE VALUES STRUCTURE
AUDI_CARS.head(20)
AUDI_CARS.describe()

#Select rows by models and some collumn names

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.info()

#check some of cdf entries
cdf.head(9)
df.head(0)

#visualize data using histogram using PANDAS
viz = cdf[["CYLINDERS","ENGINESIZE","CO2EMISSIONS","FUELCONSUMPTION_COMB"]]
viz.hist()
viz.plot()

#Visualize using scatter plots

fig = plt.figure()

plt.subplot(2,2,1) #subplot 2x2
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.subplot(2,2,2) #subplot 2x2

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="red")
plt.subplot(2,2,3) #subplot 2x2

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="black")

#TRAINING DATASET AND SPLITING

