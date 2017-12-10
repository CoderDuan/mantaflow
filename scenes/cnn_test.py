'''
slice the data by x,y,z. train in each dimension.
'''

import numpy as np
import tensorflow as tf
import sys

dim = 2
VECTOR_DIM = 3
# resolution
resC = 16
resF = 6

SEED = 12678  # Set to None for random seed.

GRID_SIZE = resC*resF+2
NUM_LABELS = GRID_SIZE*GRID_SIZE*VECTOR_DIM #10

DATA_SIZE = 800
BATCH_SIZE = 20
NUM_EPOCHS = 3000

tf.app.flags.DEFINE_boolean("self_test", True, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

# shape=[res, res, VECTOR_DIM]
def load_data(filename, shape):
    data = np.loadtxt(filename)
    # print np.shape(data)
    # data = data[0:GRID_SIZE*GRID_SIZE,:]
    # print np.shape(data)
    return data.reshape(shape)

def load_all_data(index):
    train_coarse_old_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    train_coarse_new_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    train_global_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)
    truth_data = np.ndarray(shape=(DATA_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM), dtype=np.float32)

    shape = [GRID_SIZE, GRID_SIZE, VECTOR_DIM]
    for i in range(0, DATA_SIZE):
        old_coarse_data = load_data("../data/coarse_old" + str(i+index) + ".txt",shape)
        train_coarse_old_data[i] = old_coarse_data#enlarge_data(old_coarse_data, resC, resF)

        new_coarse_data = load_data("../data/coarse"+str(i+index)+".txt", shape)
        train_coarse_new_data[i] = new_coarse_data#enlarge_data(new_coarse_data, resC, resF)

        train_global_data[i] = load_data("../data/global"+str(i+index)+".txt", shape)
        truth_data[i] = load_data("../data/groundtruth"+str(i+index)+".txt", shape)
    return train_coarse_old_data, train_coarse_new_data, train_global_data, truth_data

# enlarge the matrix data by resF*resF
def enlarge_data(data, resC, resF):
    data = data.reshape([-1, 1, VECTOR_DIM])
    tmp = np.array(data)
    for i in range(0, resF-1):
        data = np.concatenate((data, tmp), axis=1)
    data = data.reshape([-1, GRID_SIZE, VECTOR_DIM])
    tmp = np.array(data)
    for i in range(0, resF-1):
        data = np.concatenate((data, tmp), axis=1)
    data = data.reshape([-1, GRID_SIZE, VECTOR_DIM])
    return data

def slice_node_by_dim(node):
    size = [-1,-1,-1,1]
    comp_x = tf.slice(node, [0,0,0,0], size)
    comp_y = tf.slice(node, [0,0,0,1], size)
    comp_z = tf.slice(node, [0,0,0,2], size)
    return comp_x, comp_y, comp_z

def concat_3_node(a, b, c):
    node = tf.concat([a, b], 3)
    node = tf.concat([node, c], 3)
    return node

conv1_weights = tf.Variable(
    tf.truncated_normal([1, 1, 3, 1],
                        stddev=0.1,
                        seed=SEED))

def model(coarse_old_node, coarse_new_node, global_node):
    co_x, co_y, co_z = slice_node_by_dim(coarse_old_node)
    cn_x, cn_y, cn_z = slice_node_by_dim(coarse_new_node)
    gl_x, gl_y, gl_z = slice_node_by_dim(global_node)

    x_node = concat_3_node(co_x, cn_x, gl_x)
    y_node = concat_3_node(co_y, cn_y, gl_y)
    z_node = concat_3_node(co_z, cn_z, gl_z)

    conv_x = tf.nn.conv2d(x_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_y = tf.nn.conv2d(y_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_z = tf.nn.conv2d(z_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')

    predict = concat_3_node(conv_x, conv_y, conv_z)
    return predict

def stanford_model(coarse_old_node, coarse_new_node, global_node):
    return global_node + coarse_new_node - coarse_old_node

def loss_function(model, truth):
    diff = model - truth
    diff_len = tf.sqrt(tf.reduce_sum(diff*diff, 3, keep_dims=False))
    truth_len = tf.sqrt(tf.reduce_sum(truth*truth, 3, keep_dims=False) + 0.00001)
    loss = tf.reduce_mean( diff_len / truth_len ) \
            + abs(tf.reduce_sum(conv1_weights) - 1.0)
    print "loss", loss
    return loss

def loss_function2(model, truth):
    diff = model - truth
    diff_len = tf.sqrt(tf.reduce_sum(diff*diff, 3, keep_dims=False))
    truth_len = tf.sqrt(tf.reduce_sum(truth*truth, 3, keep_dims=False) + 0.00001)
    loss = tf.reduce_mean( diff*diff / (truth*truth+0.0001) ) #\
            # + abs(tf.reduce_sum(conv1_weights) - 1.0) #\
            #+ 1. / tf.reduce_sum(conv1_weights * conv1_weights + 0.001)
    print "loss", loss
    return loss

def main(argv=None):
    coarse_old_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    coarse_new_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    global_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    truth_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))

    print truth_node

    test_coarse_old_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    test_coarse_new_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    test_global_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))

    print test_coarse_old_node
    print test_coarse_new_node
    print test_global_node

    node = model(coarse_old_node, coarse_new_node, global_node)
    stanford_pred = stanford_model(coarse_old_node, coarse_new_node, global_node)
    predict = model(test_coarse_old_node, test_coarse_new_node, test_global_node)
    print "predict", predict

    loss = loss_function2(node, truth_node)
    stanford_loss = loss_function2(stanford_pred, truth_node)

    learning_rate = 0.0001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print "optimizer:",optimizer
    with tf.Session() as s:
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        # get data
        for data_index in range(0, 47):
            coarse_old_data, coarse_new_data, global_data, truth_data = load_all_data(data_index)
            print 'data loaded! Index:', DATA_SIZE

            total_step = int(NUM_EPOCHS * DATA_SIZE / BATCH_SIZE)
            for step in xrange(total_step):
                offset = (step * BATCH_SIZE) % (DATA_SIZE - BATCH_SIZE)
                # batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
                # batch_labels = train_truth_data[offset:(offset + BATCH_SIZE)]

                feed_dict = {
                    coarse_old_node:coarse_old_data[offset:(offset + BATCH_SIZE), :, :, :],
                    coarse_new_node:coarse_new_data[offset:(offset + BATCH_SIZE), :, :, :],
                    global_node:global_data[offset:(offset + BATCH_SIZE), :, :, :],
                    truth_node:truth_data[offset:(offset + BATCH_SIZE), :, :, :]
                }

                _, l, stf_l = s.run(
                    [optimizer, loss, stanford_loss],
                    feed_dict=feed_dict)
                if step % 200 == 0:
                    print tf.reshape(conv1_weights, [-1]).eval()
                    print 'Eph. %d stp. %d prog. %.2f%% loss: %.4f stf_loss: %.4f'\
                        % (float(step)*BATCH_SIZE/DATA_SIZE, step,\
                         100.*step/total_step, l, stf_l)
                    print l
                    sys.stdout.flush()

                # save the model
                if int(step+1) % 11072 == 0:
                    saver = tf.train.Saver()
                    path = saver.save(s, './cnn_test_model1/cnn_test_model1')
                    print 'Saving result to ' + path

if __name__ == '__main__':
    tf.app.run()
