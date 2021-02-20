import tensorflow as tf

x = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.int32)

y = tf.one_hot(x, depth=10)

with tf.compat.v1.Session() as sess:
    print(sess.run(y))
