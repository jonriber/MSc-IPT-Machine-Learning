#%%Importing LIBS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%GET THE DATASET and download it from the link
# get_ipython().system('wget --no-check-certificate -O FuelConsumption.csv https://cf-coursesdata.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101ENSkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv')


#%%use pandas to extract the information
df = pd.read_csv("C:\DEV\MESTRADO\MACHINE_LEARNING\FICHAS\FP03\FuelConsumptionCo2.csv")
#%%data size and dimension
np.shape(df)
np.shape(df)[0]
np.shape(df)[1]

#%%take a look at the dataset
print("HEAD")
df.head()
print("INFO")

df.info()
print("DESCRIBE")
df.describe()
df.describe()["ENGINESIZE"]
df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]].describe()
#Describe show us some usefull methods to evaluate data

#%%filtering only cars of one model, AUDI for example

mask = df["MAKE"] == "AUDI"

AUDI_CARS = df[mask] #GET ONLY TRUE VALUES STRUCTURE
AUDI_CARS.head(20)
# AUDI_CARS.head(2)
AUDI_CARS.describe()

#%%Select rows by models and some collumn names

cdf = df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.info()

#%%check some of cdf entries
cdf.head(0)
df.head(0)

#%%visualize data using histogram using PANDAS
viz = cdf[["CYLINDERS","ENGINESIZE","CO2EMISSIONS","FUELCONSUMPTION_COMB"]]
# viz = cdf[["CYLINDERS","ENGINESIZE","FUELCONSUMPTION_COMB"]]
viz.hist()
viz.plot()

#%%Visualize using scatter plots
fig = plt.figure()

plt.subplot(2,2,1) #subplot 2x2
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.subplot(2,2,2) #subplot 2x2
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="red")
plt.subplot(2,2,3) #subplot 2x2
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="black")

#%%TRAINING DATASET AND SPLITING

np.random.seed(0) #set the seed to obtain the same random values
rand_samp = np.random.rand(len(df))

plt.hist(rand_samp, bins=100)

np.random.seed(0)
msk = np.random.rand(len(df)) < 0.8 #applying a mask to get 80% ofo the values to train

# for i in range(100): print("mask",msk[i])

train = cdf[msk] #apply filter to get train samples
test = cdf[~msk] #get test samples using ~ to negate

#%%checking the result  
print("train:",train,len(train))
print("test:",test,len(test))
print("percentage", len(train)/(len(train)+len(test)))

#%%visualize training data for ENGINE SIZE VS CO2 EMISSIONS
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color= "blue")
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.show()
#%%visualize training data for CYLINDERS VS CO2 EMISSIONS
plt.scatter(train.CYLINDERS, train.CO2EMISSIONS, color= "blue")
plt.xlabel("CYLINDERS")
plt.ylabel("CO2 Emissions")
plt.show()
#%%visualize training data for FUELCONSUMPTION_COMB VS CO2 EMISSIONS
plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS, color= "blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2 Emissions")
plt.show()
#%%using sklearn package to model data
from sklearn import linear_model
from sklearn.metrics import r2_score 
regr = linear_model.LinearRegression()
#%% GETTING TRAINING DATA FOR INDEPENDENT VARIABLES
train_x_engine = np.asanyarray(train[["ENGINESIZE"]])
train_x_fuel = np.asanyarray(train[["FUELCONSUMPTION_COMB"]])
train_x_cylinder = np.asanyarray(train[["CYLINDERS"]])
# train_x = np.array(train[["ENGINESIZE"]]) ALTERNATIVE FOR ASANYARRAY
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
# print("train_y:",train_y)
#%% DEFINING GENERAL FUNCTION FOR LINEAR REGRESSION
def my_regression_function(datax, datay, label_axis_x, label_axis_y, is_testing):
    ### FITTING SECTION
    regr.fit(datax, datay)
    teta0 = regr.intercept_
    teta1 = regr.coef_
    plt.scatter(datax, datay, color="red")
    datay_ = regr.predict(datax)
    plt.plot(datax,datay,"-m",linewidth=3,color="yellow")
    plt.xlabel(label_axis_x)
    plt.ylabel(label_axis_y)
    plt.title("LINEAR REGRESSION - TRAINING DATA")
    
    ## MSE - EVALUATION ERROR SECTION 
    MSE = []
    x = df.label_axis_x
    y = df.label_axis_y
    theta_vect = np.linspace(-250,250,1000) #set theta space
    for theta in theta_vect:
        aux = 1/len(x) * np.sum((y-theta*x)**2) #MSE
        MSE.append(aux)
    
    plt.plot(theta_vect,MSE,"r",linewidth=2)
    plt.xlabel("theta")
    plt.ylabel("MSE")
    plt.title("MSE - CONVEX FUNCTION")
    
    #EVALUATION RESULT
    # ERRORS USING NUMPY FUNCTIONS
    if is_testing:
        print("TESTING EVALUATION")
        print("Mean absolute error: %.2f", np.mean(np.absolute(datay_ - datay)))
        print("Residual sum of squares (MSE): %.2f" % np.mean((datay_ - datay) ** 2))
        print("R2-score: %.2f" % r2_score(datay_ , datay) )


#%%
regr.fit(train_x, train_y)##OBTAINING COEFICIENTS OF MY LINEAR REGRESSION

# %%COEFICCIENTS
print("TETA 1", regr.coef_) #teta1
print("TETA 0", regr.intercept_) #teta0

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="red")

# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],"-r", linewidth=2, color="black")

#or another way of plotting our linear equation is using predict with independent variable x as argument

train_y = regr.predict(train_x)
plt.plot(train_x,train_y,"-m",linewidth=3,color="yellow")

plt.xlabel("ENGINE SIZE")
plt.ylabel("EMISSION")
plt.title("LINEAR REGRESSION - TRAIN")

#%% Manually calculating the predicted value of CO2 emission versus Engine Size

exResult = regr.intercept_ + (regr.coef_*(2.9))
print("EXERCISE RESULT:",exResult)

#%%MSE - CONVEX FUNCTION    
MSE = []
x = df.ENGINESIZE
y = df.CO2EMISSIONS

theta_vect = np.linspace(-100,250,1000) #set theta space

for theta in theta_vect:
    aux = 1/len(x) * np.sum((y-theta*x)**2) #MSE
    MSE.append(aux)
    
plt.plot(theta_vect,MSE,"r",linewidth=2)
plt.xlabel("theta")
plt.ylabel("MSE")
plt.title("MSE - CONVEX FUNCTION")

#%%EVALUATING OF THE FITTING

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])

#%%TEST PREDICTIONS

test_y_ = regr.predict(test_x)


plt.figure()
plt.scatter(test_x,test_y, color="blue")
plt.xlabel("ENGINE SIZE")
plt.ylabel("EMISSION")
plt.title("LINEAR REGRESSION - TEST")

plt.plot(test_x, test_y_, "r")

#%%TO-DO ANALYSE THE ERRORS!!!!!
## IS THIS MODEL GOOD OR NOT???
##  
#%% CALLING MY RECURSIVE FUNCTION
my_regression_function("FUELCONSUMPTION", "CO2EMISSIONS","FRED","FILIPE")
# my_regression_function(ENGINE_SIZE, CO2EMISSIONS)
# my_regression_function(CYLINDERS, CO2EMISSIONS)

