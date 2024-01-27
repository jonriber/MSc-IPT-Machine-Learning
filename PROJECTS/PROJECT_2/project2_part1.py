
#%% Importing libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


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
sns.countplot(y="Drug", data=df_drug)
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()

# %% GENDER DISTRIBUTION
sns.set_theme(style="darkgrid")
sns.countplot(x="Sex", data=df_drug, palette="rocket")
plt.xlabel('Gender (F=Female, M=Male)')
plt.ylabel('Total')
plt.show()

# %% BLOOD PRESSURE DISTRIBUTION
sns.set_theme(style="darkgrid")
sns.countplot(y="BP", data=df_drug, palette="crest")
plt.ylabel('Blood Pressure')
plt.xlabel('Total')
plt.show()

# %% Cholesterol DISTRIBUTION
sns.set_theme(style="darkgrid")
sns.countplot(x="Cholesterol", data=df_drug, palette="magma")
plt.xlabel('Blood Pressure')
plt.ylabel('Total')
plt.show()

# %% Gender Distribution based on Drug Type

pd.crosstab(df_drug.Sex,df_drug.Drug).plot(kind="bar",figsize=(12,5),color=['#003f5c','#ffa600','#58508d','#bc5090','#ff6361'])
plt.title('Gender distribution based on Drug type')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

#%% Blood Pressure Distribution based on Cholesetrol
pd.crosstab(df_drug.BP,df_drug.Cholesterol).plot(kind="bar",figsize=(15,6),color=['#6929c4','#1192e8'])
plt.title('Blood Pressure distribution based on Cholesterol')
plt.xlabel('Blood Pressure')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

#%% Sodium to Potassium Distribution based on Gender and Age

plt.scatter(x=df_drug.Age[df_drug.Sex=='F'], y=df_drug.Na_to_K[(df_drug.Sex=='F')], c="Blue")
plt.scatter(x=df_drug.Age[df_drug.Sex=='M'], y=df_drug.Na_to_K[(df_drug.Sex=='M')], c="Orange")
plt.legend(["Female", "Male"])
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.show()

#%% DATA SET PREPARATION

## AGE CATEGORY
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis = 1)

#%% Na_to_K
bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis = 1)


#%% Splitting the dataset

X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#%% Feature Engineering

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train.head()

X_test.head()

#%% SMOTE Technique
smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)

sns.set_theme(style="darkgrid")
sns.countplot(y=y_train, data=df_drug, palette="mako_r")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()

#%% MODELS

## LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))

#%% 
# K Neighbours
from sklearn.neighbors import KNeighborsClassifier
KNclassifier = KNeighborsClassifier(n_neighbors=20)
KNclassifier.fit(X_train, y_train)

y_pred = KNclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
KNAcc = accuracy_score(y_pred,y_test)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc*100))

#%% 
# SVM
from sklearn.svm import SVC
SVCclassifier = SVC(kernel='linear', max_iter=251)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))


#%% PIPELINE TIME!

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming X_train, X_test, y_train, y_test are already defined

# Create a pipeline with a preprocessing step (standardization) and the logistic regression classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('classifier', LogisticRegression(solver='liblinear', max_iter=5000))  # Logistic regression classifier
])

# Define the hyperparameters to search
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'classifier__penalty': ['l1', 'l2']  # Regularization penalty
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(X_train, y_train)

# Make predictions using the best model found by grid search
y_pred = grid_search.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
LRAcc = accuracy_score(y_pred, y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc * 100))

# Get the best hyperparameters found by grid search
print("Best hyperparameters:", grid_search.best_params_)

#%% PIPELINE WITH 5 DIFFERENT MODELS
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define the models and their corresponding parameters
models = {
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', max_iter=5000),
        'params': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2']
        }
    },
    'k_neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [3, 5, 7, 9]
        }
    },
    'svm': {
        'model': SVC(),
        'params': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf']
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'classifier__max_depth': [3, 5, 7, 9]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7]
        }
    }
}

# Create an empty dictionary to store the results for each model
results = {}

# Loop through the models and perform grid search for each model
for model_name, mp in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', mp['model'])
    ])
    grid_search = GridSearchCV(pipeline, mp['params'], cv=5)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    results[model_name] = {'model': grid_search, 'accuracy': accuracy}

# Print the results for each model
model_comparison = []
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(classification_report(y_test, result['model'].predict(X_test)))
    print(confusion_matrix(y_test, result['model'].predict(X_test)))
    print(f"{model_name} accuracy is: {result['accuracy']*100:.2f}%")
    print(f"Best hyperparameters for {model_name}: {result['model'].best_params_}")
    print("\n")
    model_comparison.append({'Model': model_name, 'Accuracy': result['accuracy']*100})

# %%
import pandas as pd

# Initialize an empty list to store the model names and accuracies
df_model_comparison = pd.DataFrame(model_comparison)

# Print the model comparison dataframe
print(df_model_comparison)
# %%
