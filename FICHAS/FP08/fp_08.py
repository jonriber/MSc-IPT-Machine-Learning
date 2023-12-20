

#%% Importing MODULES
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.datasets import fetch_openml

#%%MNIST DATASET - EXERCISE 1
mnist = fetch_openml('mnist_784', version=1, as_frame = False) #as_frame = False otherwise
#returns data as a dataframe
mnist.keys()
mnist.url ##https://www.openml.org/d/554
mnist.DESCR

#%% DATA SHAPE
X, y = mnist["data"], mnist["target"]
X.shape
y.shape

some_digit = X[0] #first
print("some_digit:",some_digit)
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary) #binay (grayscale)
plt.axis("off")
y[0] #label

#transform string labels into int8
y = y.astype(np.uint8) #or y=np.uint8(y)
y[0]


#%% FUNCTION TO PLOT DIGITS

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
        interpolation="nearest")
    plt.axis("off")

#%%TESTING THE FUNCTION
plot_digit(X[1])
plot_digit(X[0])
plot_digit(X[5])
print("wX[5]:",X[5])
print("y[5]",y[5])

# %% LOAD ALL DATASET (TRAINING AND TESTTING)

X_train, X_test, y_train, y_test = X[:60000],X[60000:], y[:60000],y[60000:]

#%% training a binary classifier
# SET ClASS 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# print(y_train_5)

#%% FOR loop
for i in range(np.shape(y_train)[0]):
    if (y_train[i] == True) : print(i)
    
#%% CREATING A MODEL

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# sgd_clf.predict([some_digit])
sgd_clf.predict([X[0]])
sgd_clf.score(X_train, y_train_5)

#or
#%% USING PREDICT
y_predict = sgd_clf.predict(X_train)
acc_score = np.mean(y_predict == y_train_5)
acc_score
# %% USING ACCURACY SCORE
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_predict, y_train_5)
acc_score

# %% 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
accuracy_score(y_train_5, y_predict)
balanced_accuracy_score(y_train_5, y_predict)
precision_score(y_train_5, y_predict)
recall_score(y_train_5, y_predict)
f1_score(y_train_5, y_predict)
#%% GET ALL METRICS FROM SKLEARN
from sklearn.metrics import get_scorer_names
get_scorer_names()

#%% CHECKING NUMBER OF 5 and non-5
n5 = np.sum(y_train_5 == True)
print(n5)
#number of non-5
n_not5 = np.sum(y_train_5 == False)
n_not5

#%%CROSS VALIDATION =3 FOLDS, TABLE COLUMN 2
from sklearn.model_selection import cross_val_score, cross_val_predict
scores=cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="balanced_accuracy")
scores
np.mean(scores)


#or obtain using prediction and evaluation score
y_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
balanced_accuracy_score(y_train_5,y_predict)

accuracy_score(y_train_5,y_predict)
balanced_accuracy_score(y_train_5,y_predict)
precision_score(y_train_5,y_predict)
recall_score(y_train_5,y_predict)
f1_score(y_train_5,y_predict)

#%%GRIDSEARCHCV

# TO DO: TRY to find what works the most

from sklearn.model_selection import GridSearchCV
param_grid = [
 {'alpha': [10, 1, 0.1, 0.001], 'eta0': [0, 0.01, 0.002]},
 ]
sgd_clf_new = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
grid_search = GridSearchCV(sgd_clf_new, param_grid, cv=3,scoring=('f1'))

grid_search.fit(X_train, y_train_5)
grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
grid_search.best_estimator_ #best model!

#%%BEST MODEL with test data (X_test)
sgd_clf_best = SGDClassifier(alpha=10, eta0=0, random_state=42)
sgd_clf_best.fit(X_train, y_train_5)
y_predict = cross_val_predict(sgd_clf_best, X_train, y_train_5, cv=3)
# or
y_predict = cross_val_predict(grid_search.best_estimator_, X_train,y_train_5, cv=3)

#RESULT OF COLUMN 3
balanced_accuracy_score(y_train_5,y_predict)
accuracy_score(y_train_5,y_predict)
balanced_accuracy_score(y_train_5,y_predict)
precision_score(y_train_5,y_predict)
recall_score(y_train_5,y_predict)
f1_score(y_train_5,y_predict)

#%% RESULT OF COLUMN 4
# REAL WORLD SCENARIO
y_predict = grid_search.best_estimator_.predict(X_test)

balanced_accuracy_score(y_test_5,y_predict)
accuracy_score(y_test_5,y_predict)
balanced_accuracy_score(y_test_5,y_predict)
precision_score(y_test_5,y_predict)
recall_score(y_test_5,y_predict)
f1_score(y_test_5,y_predict)

#%% CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
y_predict = grid_search.best_estimator_.predict(X_train)

confusion_matrix(y_train_5, y_predict)
#precision of class 5
precision_score(y_train_5, y_predict)
precision = (4403)/(4403 + 351) #TP/(TP+FP)
precision


#%%CALCULATE RECALL THE SAME WAY
recall_score(y_train_5,y_predict)
recall = (4403)/(4403+1018) # TP /TP+FN
recall

#%% DECISION FUNCTION
#One digit only
some_digit = X[0]
y_score = sgd_clf.decision_function([some_digit])
y_score
#ALL DIGITS
y_scores_5 = sgd_clf.decision_function(X_train[y_train_5,:])
y_scores_not_5 = sgd_clf.decision_function(X_train[~y_train_5,:])
fig = plt.figure()
plt.hist(y_scores_5, bins='auto', color='b', alpha=0.5)
plt.hist(y_scores_not_5, bins='auto', color='r', alpha=0.5)
fig.savefig("Histogram_prob.png")
# %%PRECISION RECALL ROC

from sklearn.metrics import precision_recall_curve
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
precisions
recalls
thresholds

#%% plot precision recall thresholds function
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1]) 

#%% Ploting ROC
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# %%PRECISION AGAINST THE RECALL FUNCTION

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

#%% USING THE PLOT PRECISION VS RECALL FUNCTION
fig=plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
fig.savefig("precision_vs_recall_plot.png")

#%%PLOT the ROC curve and AUC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

#%% Defininf plot_roc_curve function
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True) 

#%% USING the plot_roc_curve function
fig= plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
fig.savefig("ROC.png")
#%% 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#%% MULTICLASS - MOST USED FOR MORE THAN 2 CLASSES
# IN THIS CASE, 10 CLASSES

mult_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
mult_clf.fit(X_train, y_train) # SGD applies OnevsAll
y_predict = mult_clf.predict(X_train)

accuracy_score(y_train,y_predict)
balanced_accuracy_score(y_train, y_predict)
precision_score(y_train, y_predict, average='macro')
recall_score(y_train, y_predict, average='macro')
f1_score(y_train, y_predict, average='macro')

#%%CONFUSION MATRIX FOR MULTICLASS
y_predict = cross_val_predict(mult_clf, X_train, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_predict)
plt.figure(figsize=(8,8))
plt.matshow(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)

#%% DEFINING PLOT CONFUSION MATRIX FUNCTION
def plot_confusion_matrix(matrix):
    #with color bar
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix) #, cmap=plt.cm.gray)
    fig.colorbar(cax)

#%% USING plot_confusion_matrix
plot_confusion_matrix(conf_mx)


#%% CHECK ERROR RATE (NORMALIZATION)
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

#%%
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_predict == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_predict == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_predict == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_predict == cl_b)]

#%%plot_digits function definition
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

#%% USING PLOT DIGITS FUNCTION

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

#%% StratifiedKFold t
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)


#%% for loop
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    print(np.mean(y_pred == y_test_fold))
    #or
    #print(accuracy_score(y_test_fold, y_pred))