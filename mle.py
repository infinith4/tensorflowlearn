import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame,Series

np.random.seed(20160512)
n0,mu0,variance0 = 20,[10,11],20
data0 = multivariate_normal(mu0,np.eye(2)*variance0,n0)
df0 = DataFrame(data0,columns=['x1','x2'])
df0['t'] = 0

n1,mu1,variance1 = 15,[18,20],22
data1 = multivariate_normal(mu1,np.eye(2)*variance1,n1)
df1 = DataFrame(data1,columns=['x1','x2'])
df1['t'] = 1


df = pd.concat([df0,df1],ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)
