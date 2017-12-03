import tensorflow as tf

matrix1 = tf.constant([[1., 2., 3.], [1., 2., 3.]])

matrix2 = tf.constant([[3., 2., 1.], [1., 2., 3.]])

product = tf.div(matrix1, matrix2)

sess = tf.Session()

print sess.run(product)