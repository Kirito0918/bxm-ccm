import tensorflow as tf

a = tf.range(10)*5

with tf.Session() as sess:
    print(sess.run(a))