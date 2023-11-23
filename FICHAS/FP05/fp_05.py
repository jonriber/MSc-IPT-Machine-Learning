#%% FP 05 - Batch gradient Descent (BGD)

#importing libraries
import matplotlib.pyplot as plt
import numpy as np

#%% Exercise 1
#generating synthetic data
np.random.seed(42)

X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.rand(100,1)

plt.figure(figsize=(6,4))

plt.scatter(X,y)


#%% Exercise 2
#creating the function
def batch_gradient_descent(X,y,n_iterations, learning_rate, all_path):
    np.random.seed(42)
    thetas = np.random.randn(2, 1) #random initialization
    thetas_path = [thetas]
    m = np.shape(X)[0]
    X_b = np.concatenate([np.ones((m, 1)) , X], axis=1)
    for i in range(n_iterations):
        gradients = 2*np.transpose(X_b) @ (X_b @ (thetas) -y)/m
        thetas = thetas - learning_rate*gradients
        if all_path == 1:
            thetas_path.append(thetas) #all iterations
        else:
            thetas_path = thetas

    return thetas_path

n_iterations = 1000
learning_rate = 0.1

theta_path_bgd = batch_gradient_descent(X,y,n_iterations, learning_rate,1)
theta_path_bgd #all iterations

dim = np.shape(theta_path_bgd)
theta_est = theta_path_bgd[dim[0]-1][:]
theta_est

# %%

y_est = X @ theta_est[1:] + theta_est[0]
plt.plot(X, y_est, "-r", linewidth=3)
plt.scatter(X,y, color="b")


#%%
from sklearn.metrics import r2_score

print("r2_score", r2_score(y,y_est))
# %% Exercise 3 defining function 


