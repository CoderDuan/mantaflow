import tensorflow as tf
import numpy as np

SEED = 233
BATCH_SIZE = 1

dim = 2
VECTOR_DIM = 3
# resolution
resC = 6
resF = 6

DATA_SIZE = 39
VALIDATION_SIZE = 9
NUM_EPOCHS = 100

# shape=[res, res, VECTOR_DIM]
def load_data(filename, shape):
	return np.loadtxt(filename).reshape(shape)

def load_all_data():
	train_coarse_old_data = np.ndarray(shape=(DATA_SIZE, resC*resF, resC*resF, VECTOR_DIM), dtype=np.float32)
	train_coarse_new_data = np.ndarray(shape=(DATA_SIZE, resC*resF, resC*resF, VECTOR_DIM), dtype=np.float32)
	train_global_data = np.ndarray(shape=(DATA_SIZE, resC*resF, resC*resF, VECTOR_DIM), dtype=np.float32)
	truth_data = np.ndarray(shape=(DATA_SIZE, resC*resF, resC*resF, VECTOR_DIM), dtype=np.float32)
	shape = [resC*resF, resC*resF, VECTOR_DIM]
	for i in range(1, DATA_SIZE+1):
		old_coarse_data = load_data("../data/coarse" + str(i-1) + ".txt", [resC, resC, VECTOR_DIM])
		train_coarse_old_data[i-1] = enlarge_data(old_coarse_data, resC, resF)

		new_coarse_data = load_data("../data/coarse"+str(i)+".txt", [resC, resF, VECTOR_DIM])
		train_coarse_new_data[i-1] = enlarge_data(new_coarse_data, resC, resF)

		train_global_data[i-1] = load_data("../data/global"+str(i)+".txt", shape)
		truth_data[i-1] = load_data("../data/groundtruth"+str(i)+".txt", shape)
	return train_coarse_old_data, train_coarse_new_data, train_global_data, truth_data

# enlarge the matrix data by resF*resF
def enlarge_data(data, resC, resF):
	data = data.reshape([-1, 1, VECTOR_DIM])
	tmp = np.array(data)
	for i in range(0, resF-1):
		data = np.concatenate((data, tmp), axis=1)
	data = data.reshape([-1, resC*resF, VECTOR_DIM])
	tmp = np.array(data)
	for i in range(0, resF-1):
		data = np.concatenate((data, tmp), axis=1)
	data = data.reshape([-1, resC*resF, VECTOR_DIM])
	return data

def model(node, resC, resF):
	conv1_weights = tf.Variable(tf.truncated_normal([5, 5, VECTOR_DIM*3, 16], stddev=0.1, seed=SEED))
	# conv1_biases = tf.Variable(tf.zeros([16]))

	conv = tf.nn.conv2d(node, conv1_weights, strides=[1,1,1,1], padding='SAME')
	# active = tf.nn.tanh(tf.nn.bias_add(conv, conv1_biases))

	conv2_weights = tf.Variable(
		tf.truncated_normal([5, 5, 16, VECTOR_DIM],
							stddev=0.1,
							seed=SEED))
	# conv2_biases = tf.Variable(tf.zeros([VECTOR_DIM]))
	conv = tf.nn.conv2d(conv, conv2_weights, strides=[1,1,1,1], padding='SAME')
	return conv

def loss_function(test, truth):
	test = tf.reshape(test, [1,-1])
	truth = tf.reshape(truth, [1,-1])
	return tf.reduce_sum(tf.square(test - truth))

def main(argv=None):
	train_coarse_old_data, train_coarse_new_data, train_global_data, train_truth_data = load_all_data()
	
	train_coarse_old_data = train_coarse_old_data[VALIDATION_SIZE:,:,:,:]
	train_coarse_new_data = train_coarse_new_data[VALIDATION_SIZE:,:,:,:]
	train_global_data = train_global_data[VALIDATION_SIZE:,:,:,:]
	train_truth_data = train_truth_data[VALIDATION_SIZE:,:,:,:]

	validation_old_data = train_coarse_old_data[:VALIDATION_SIZE,:,:,:]
	validation_new_data = train_coarse_new_data[:VALIDATION_SIZE,:,:,:]
	validation_global_data = train_global_data[:VALIDATION_SIZE,:,:,:]
	validation_truth_data = train_truth_data[:VALIDATION_SIZE,:,:,:]

	start = tf.placeholder(tf.float32, shape=(BATCH_SIZE, resC*resF, resC*resF, VECTOR_DIM*3))
	gtruth = tf.placeholder(tf.float32, shape=(BATCH_SIZE, resC*resF, resC*resF, VECTOR_DIM))
	node = model(start, resC, resF)

	loss = loss_function(node, gtruth)
	train_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	train_size = train_truth_data.shape[0]

	validation_data = np.concatenate((validation_global_data, validation_new_data), axis=3)
	validation_data = np.concatenate((validation_data, validation_old_data), axis=3)
	validation_node = tf.constant(validation_data)
	validation_truth_node = tf.constant(validation_truth_data)
	validation_loss = loss_function(model(validation_node, resC, resF), validation_truth_node)

	with tf.Session() as s:
		s.run(tf.initialize_all_variables())
		for i in xrange(int(NUM_EPOCHS * train_size / BATCH_SIZE)):
			offset = (i * BATCH_SIZE) % (train_size - BATCH_SIZE)
			
			batch_old_data = train_coarse_old_data[offset:(offset+BATCH_SIZE), :, :, :]
			batch_new_data = train_coarse_new_data[offset:(offset+BATCH_SIZE), :, :, :]
			batch_global_data = train_global_data[offset:(offset+BATCH_SIZE), :, :, :]
			batch_truth_data = train_truth_data[offset:(offset+BATCH_SIZE), :, :, :]

			batch_data = np.concatenate((batch_global_data, batch_new_data), axis=3)
			batch_data = np.concatenate((batch_data, batch_old_data), axis=3)

			feed_dict = {start:batch_data, gtruth:batch_truth_data}

			s.run(train_step, feed_dict=feed_dict)
			if i % 10 == 0:
				validation_result = 0
				for j in xrange(int(VALIDATION_SIZE/BATCH_SIZE)):
					validation_result += validation_loss.eval()
				print "validation_result:", validation_result
			
if __name__ == '__main__':
    tf.app.run()
# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([0.100, 0.200], x_data) + 0.300

# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b

# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()

# sess = tf.Session()
# sess.run(init)

# for step in xrange(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print step, sess.run(W), sess.run(b)