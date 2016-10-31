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

train_x = train_set[['x1','x2']].as_matrix()
train_t = train_set['t'].as_matrix().reshape([len(train_set),1])
x = tf.placeholder(tf.float32,[None,2])
w = tf.Variable(tf.zeros([2,1]))
w0 = tf.Variable(tf.zeros([1]))
f = tf.matmul(x,w) + w0
p = tf.sigmoid(f)

t = tf.placeholder(tf.float32,[None,1])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)+tf.log(1-p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5),tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(0,20000):
    i+=1
    sess.run(train_step,feed_dict={x:train_x,t:train_t})
    if i % 2000 == 0:
        loss_val,acc_val = sess.run([loss,accuracy],feed_dict={x:train_x,t:train_t})
        print('Step:%d,Loss:%f,Accuracy:%f' % (i,loss_val,acc_val))
# Step:2000,Loss:7.463896,Accuracy:0.600000
# Step:4000,Loss:5.210213,Accuracy:0.685714
# Step:6000,Loss:4.460924,Accuracy:0.685714
# Step:8000,Loss:4.315311,Accuracy:0.685714
# Step:10000,Loss:4.309548,Accuracy:0.685714
# Step:12000,Loss:4.309571,Accuracy:0.685714
# Step:14000,Loss:4.309546,Accuracy:0.685714
# Step:16000,Loss:4.309547,Accuracy:0.685714
# Step:18000,Loss:4.309546,Accuracy:0.685714
# Step:20000,Loss:4.309547,Accuracy:0.685714