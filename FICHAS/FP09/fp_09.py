#%% IMPORTING LIBS
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score,precision_score, f1_score
import matplotlib.pyplot as plt

#%% 
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris virginica

#%% PIPELINE
svm_clf = Pipeline([
 ("scaler", StandardScaler()),
 ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
 ])

svm_clf.fit(X, y)
#%% Predict

y_predict = svm_clf.predict(X)
y_train = y

#%% EVALUATION

accuracy_score(y_train,y_predict)
balanced_accuracy_score(y_train, y_predict)
precision_score(y_train, y_predict)
recall_score(y_train, y_predict)
f1_score(y_train, y_predict)

#%% Applying scaling
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42, max_iter=20000)

scaled_svm_clf1 = Pipeline([
 ("scaler", scaler),
 ("linear_svc", svm_clf1),
 ])

scaled_svm_clf2 = Pipeline([
 ("scaler", scaler),
 ("linear_svc", svm_clf2),
 ])

#%% Two fits, one for each model
scaled_svm_clf1.fit(X, y) 
scaled_svm_clf2.fit(X, y)

#%% PREDICT for first Model and second Model
y_predict = scaled_svm_clf1.predict(X)
balanced_accuracy_score(y, y_predict)
y_predict = scaled_svm_clf2.predict(X)
balanced_accuracy_score(y_train, y_predict)


#%% GRID SEARCH

from sklearn.model_selection import GridSearchCV
param_grid = [
 {'C': [0.1, 1, 10, 100]},
 ]

#%%
svm_clf_grid = LinearSVC(loss="hinge", random_state=42, max_iter=100000)
grid_search = GridSearchCV(svm_clf_grid, param_grid, cv=3,scoring=('balanced_accuracy'))

#%%
grid_search.fit(X, y)
grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
grid_search.best_estimator_
y_est=grid_search.best_estimator_.predict(X)
balanced_accuracy_score(y, y_est)

#%% #GRIDSEARCH with PIPELINE
#In case of using pipeline in GridSearchCV it is required to use in paramgrid 'step_name__parameter_name'
#the double underscore (__) is used in scikit-learn to indicate nesting or specifying parameters within a
#pipeline
#in this particular case 'linear_svc__C'

param_grid = [
 {'linear_svc__C': [0.1, 1, 10, 100]},
 ]

svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42, max_iter=20000)

scaled_svm_clf2 = Pipeline([
 ("scaler", scaler),
 ("linear_svc", svm_clf2),
 ])

grid_search = GridSearchCV(scaled_svm_clf2, param_grid, cv=3,
 scoring=('balanced_accuracy'))

#%% BEST SCORES
grid_search.fit(X, y)
grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
grid_search.best_estimator

# y_est = grid_search.best_estimator_.predict(X)
# balanced_accuracy_score(y, y_est)


#%% #dataset MakeMoons #################
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

#%%
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
 
#%%
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

#%%
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
 ("poly_features", PolynomialFeatures(degree=3)),
 ("scaler", StandardScaler()),
 ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42, max_iter=20))
 ])
polynomial_svm_clf.fit(X, y)

#%%
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

#%%
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

#%%
y_predict = polynomial_svm_clf.predict(X)
accuracy_score(y, y_predict)
balanced_accuracy_score(y, y_predict)
precision_score(y, y_predict)
recall_score(y, y_predict)
f1_score(y, y_predict)

#%% EXERCISE 3
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
 ("scaler", StandardScaler()),
 ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
 ])

poly_kernel_svm_clf.fit(X, y)


y_predict = poly_kernel_svm_clf.predict(X)
accuracy_score(y, y_predict)
balanced_accuracy_score(y, y_predict)
precision_score(y, y_predict)
recall_score(y, y_predict)
f1_score(y, y_predict)