# exercise 14
# 3 dimensional data 
# data set from  https://www.dropbox.com/s/xg2g6c1x5pw3njc/P300_LSC_dataset.mat?dl=0
import scipy.io as sc
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


data = sc.loadmat("C:\DEV\MESTRADO\MACHINE_LEARNING\FICHAS\FP02\P300_LSC_dataset.mat")
data.keys()
data1 = data['P300_LSC_dataset'] #structure with target and non-targets
data1.dtype
yt = data1[0,0]['ytarget']; #ytarget: channels x time sample x trials
ynt = data1[0,0]['yNONtarget'];
np.shape(yt)
np.shape(ynt)
#the dataset is very imbalanced, so we may limit NONtarget trials to fewer trials , let's
#say 840
ynt = ynt[:,:, 0:840]

label1='P300'
label2='Standard'

print('P300 DATASET\n')
print("Variable yt: P300 epochs - %d channels x %d time samples x %d target trials\n",
    np.size(yt,0),np.size(yt,1),np.size(yt,2));
print("Variable ynt: standard epochs - %d channels x %d time samples x %d NONtarget trials\n", 
    np.size(ynt,0),np.size(ynt,1),np.size(ynt,2));

#%%
fs = 256
ts = 1/fs

#%%
t = np.arange(fs) * ts

#%%
yt_mean = np.mean(yt, axis=2)
yt_std = np.std(yt,axis=2)

ynt_mean = np.mean(ynt, axis=2)
ynt_std = np.std(ynt, axis=2)

np.shape(yt_mean)

#%%

ch = 1
plt.figure()
plt.plot(t, yt_mean[1:],"r",label=label1)
plt.plot(t, ynt_mean[1:],"b",label2)
plt.xlabel("time(s)")
plt.ylabel("amplitude (uV)")
plt.legend(loc = "upper left")






# %%
