# 02_multi_class_logistic_regression_tensorflow

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(0)

M = 2      # 入力データの次元
K = 3      # クラス数
n = 100    # クラスごとのデータ数
N = n * K  # 全データ数

'''
データの生成
'''
X1 = np.random.randn(n, M) + np.array([0, 10]) #nxM
X2 = np.random.randn(n, M) + np.array([5, 5]) #nxM
X3 = np.random.randn(n, M) + np.array([10, 0]) #nxM
Y1 = np.array([[1, 0, 0] for i in range(n)]) # 1〜n行が[1, 0, 0]のn×3の行列
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

print(Y1)
X = np.concatenate((X1, X2, X3), axis=0) #3nxM axis=0は縦に連結
Y = np.concatenate((Y1, Y2, Y3), axis=0) #3nxM
print(np.shape(X)) #300x2
print(np.shape(Y)) #300x2

'''
モデル設定
'''
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y),
                               reduction_indices=[1])) #reduction_indices は結果を1次元に集約
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
argmax_y = tf.argmax(y, 1)
argmax_t = tf.argmax(t, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

'''
モデル学習
'''
# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 50  # ミニバッチサイズ
n_batches = N // batch_size  # 整数対整数による除算
print(n_batches)

# ミニバッチ学習
for epoch in range(20): # [0,1,...,19]
    X_, Y_ = shuffle(X, Y)

    for i in range(n_batches):   #6 batch
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        }) #feed_dict:placeholderをkey、tensorをvlaueとしたdictionaryを作る。

'''
学習結果の確認
'''
X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)

sess_W = sess.run(W)
sess_b = sess.run(b)
print(sess_W)
print(sess_b)
