http://dev.classmethod.jp/machine-learning/tensorflow-math/


tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)

損失関数などで使用する平均を算出する関数
reduction_indicesで集約する次元を指定する

実行サンプル
>>> a = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]])

>>> mean = tf.reduce_mean(a)
>>> sess.run(mean)
3.5

>>> mean = tf.reduce_mean(a,0)
>>> sess.run(mean)
array([ 3.,  4.], dtype=float32)

>>> mean = tf.reduce_mean(a,1)
>>> sess.run(mean)
array([ 1.5,  3.5,  5.5], dtype=float32)
