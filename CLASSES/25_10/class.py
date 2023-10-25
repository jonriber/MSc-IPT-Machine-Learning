

from sklearn import linear_model

#DEFININF TYPE OF MODEL
regr = linear_model.LinearRegression()
# FITTING
regr.fit(train_x, train_y)
#PREDICTING
test_y_ = regr.predict(test_y_)