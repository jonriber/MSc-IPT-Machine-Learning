

from sklearn import linear_model

#DEFININF TYPE OF MODEL
regr = linear_model.LinearRegression()
# FITTING
regr.fit(train_x, train_y)
#PREDICTING
test_y_ = regr.predict(test_y_)

#METRICS

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# where test_y is the ground proof or real value
# where test_y_ is the estimated value
r2 = r2_score(test_y, test_y_)
mse = mean_squared_error(test_y, test_y_)
rmse = mean_squared_error(test_y, test_y_,squared=False)
mae = mean_absolute_error(test_y_, test_y)