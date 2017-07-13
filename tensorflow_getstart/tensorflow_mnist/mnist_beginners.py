from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
# placeholder はデータの入れ物。モデルの定義の時は、データの次元だけ決めておく。
# モデルの学習など、実際のデータが必要になった時、値を入れる。
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10])) #変数の生成 W:784*10
b = tf.Variable(tf.zeros([10])) #10
print(W)

y = tf.nn.softmax(tf.matmul(x, W) + b) #xW＋ｂ
print(y)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
 #「ある数値の軍団Aと、ある数値の軍団Bがどれくらい異なるか」を表す概念です。エントロピーとは、これもざっくり述べますと「乱雑さの度合い」です。この「エントロピー」という概念は様々な理工学分野で用いられまして、上下に分離したドレッシングの瓶をシャカシャカ振ってよく混ざった状態にすると「エントロピーが増えた」となります。

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy) #learning rate:0.05. minimize:cross_entropy

sess = tf.InteractiveSession()
#create an operation to initialize the variables
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) #we get a "batch" of one hundred random data points from our training set. 
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
