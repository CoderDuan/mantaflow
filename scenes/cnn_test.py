"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.
"""
import gzip
import os
import sys
import urllib

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

dim = 2
VECTOR_DIM = 3
# resolution
resC = 8
resF = 4

SEED = 66478  # Set to None for random seed.

GRID_SIZE = resC*resF+2
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = GRID_SIZE*GRID_SIZE*VECTOR_DIM #10

DATA_SIZE = 100
VALIDATION_SIZE = 0#5000  # Size of the validation set.
BATCH_SIZE = 1
NUM_EPOCHS = 3000

tf.app.flags.DEFINE_boolean("self_test", True, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

# shape=[res, res, VECTOR_DIM]
def load_data(filename, shape):
    return np.loadtxt(filename).reshape(shape)

def load_all_data():
    train_coarse_old_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    train_coarse_new_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    train_global_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    truth_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)

    shape = [GRID_SIZE, GRID_SIZE, VECTOR_DIM]
    for i in range(0, DATA_SIZE):
        old_coarse_data = load_data("../data/coarse_old" + str(i) + ".txt", shape)
        train_coarse_old_data[i] = old_coarse_data#(old_coarse_data, resC, resF)

        new_coarse_data = load_data("../data/coarse"+str(i)+".txt", shape)
        train_coarse_new_data[i] = new_coarse_data#(new_coarse_data, resC, resF)

        train_global_data[i] = load_data("../data/global"+str(i)+".txt", shape)
        truth_data[i] = load_data("../data/groundtruth"+str(i)+".txt", shape)
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

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
    """The Model definition."""
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, VECTOR_DIM*3, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 3],
                            stddev=0.1,
                            seed=SEED))
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # conv = tf.nn.dropout(conv, 0.95, seed=SEED)
    conv = tf.nn.conv2d(conv,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    print conv

    return conv

def loss_function(model, truth):
    # model = tf.reshape(model, [BATCH_SIZE, NUM_LABELS])
    # truth = tf.reshape(truth, [BATCH_SIZE, NUM_LABELS])
    loss = tf.reduce_mean(tf.div(tf.abs(model - truth), (tf.abs(truth) + 0.001)))
    return loss

def loss_function2(model, truth):
    truth = tf.reshape(truth, [BATCH_SIZE, NUM_LABELS])
    loss = tf.nn.softmax(tf.abs(model - truth))

def main(argv=None):  # pylint: disable=unused-argument
    # get data
    coarse_old_data, coarse_new_data, global_data, truth_data = load_all_data()
    print "data loaded!"

    train_coarse_old_data = coarse_old_data[VALIDATION_SIZE:,:,:,:]
    train_coarse_new_data = coarse_new_data[VALIDATION_SIZE:,:,:,:]
    train_global_data = global_data[VALIDATION_SIZE:,:,:,:]
    train_data = np.concatenate((train_global_data, train_coarse_new_data), axis=3)
    train_data = np.concatenate((train_data, train_coarse_old_data), axis=3)
    train_truth_data = truth_data[VALIDATION_SIZE:,:,:,:]
    # train_truth_data = train_truth_data.reshape([-1, resC*resF, resC*resF, VECTOR_DIM])

    validation_old_data = coarse_old_data[:VALIDATION_SIZE,:,:,:]
    validation_new_data = coarse_new_data[:VALIDATION_SIZE,:,:,:]
    validation_global_data = global_data[:VALIDATION_SIZE,:,:,:]
    validation_data = np.concatenate((validation_global_data, validation_new_data), axis=3)
    validation_data = np.concatenate((validation_data, validation_old_data), axis=3)
    validation_truth_data = truth_data[:VALIDATION_SIZE,:,:,:]
    validation_truth_data = validation_truth_data.reshape([-1, NUM_LABELS])

    num_epochs = NUM_EPOCHS

    train_size = train_truth_data.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM*3))
    # train_truth_node = tf.placeholder(tf.float32,
    #                                    shape=(BATCH_SIZE, NUM_LABELS))

    train_truth_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    validation_data_node = tf.constant(validation_data)
    
    # Training computation: node + cross-entropy loss.
    node = model(train_data_node, True)
    print "node:", node
    print node.get_shape()

    loss = loss_function(node, train_truth_node)

    learning_rate = 0.00001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print "optimizer:",optimizer

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_truth_data[offset:(offset + BATCH_SIZE), :, :, :]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_truth_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l = s.run(
                [optimizer, loss],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %.3f, Minibatch loss: %.6f' % (float(step) * BATCH_SIZE / train_size, l)
                sys.stdout.flush()

            # save the model
            if int(step+1) % 2000 == 0:
                saver = tf.train.Saver()
        	path = saver.save(s, './save/cnn_test_model')
		print 'Saving result to ' + path

if __name__ == '__main__':
    tf.app.run()
