import numpy as np
import tensorflow as tf
import sys

dim = 2
VECTOR_DIM = 3
# resolution
resC = 8
resF = 4

SEED = 66478  # Set to None for random seed.

GRID_SIZE = resC*resF+2
NUM_LABELS = GRID_SIZE*GRID_SIZE*VECTOR_DIM #10

DATA_SIZE = 100
BATCH_SIZE = 10
NUM_EPOCHS = 30000

tf.app.flags.DEFINE_boolean("self_test", True, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

# shape=[res, res, VECTOR_DIM]
def load_data(filename, shape):
    return np.loadtxt(filename).reshape(shape)

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

def model(coarse_old_node, coarse_new_node, global_node):
    alpha = tf.Variable(0.5, name='alpha')
    # predict = alpha * global_node + (1. - alpha) * (coarse_new_node - coarse_old_node)
    predict = alpha*coarse_new_node + (1.-alpha)*(global_node + coarse_new_node - coarse_old_node)
    print "predict", predict
    return predict, alpha

def loss_function(model, truth):
    loss = tf.reduce_sum(tf.abs(model - truth)) / tf.reduce_sum(tf.abs(truth))
    print loss
    return loss

def loss_function2(model, truth):
    truth = tf.reshape(truth, [BATCH_SIZE, NUM_LABELS])
    loss = tf.nn.softmax(tf.abs(model - truth))

def main(argv=None):
    coarse_old_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    coarse_new_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    global_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    truth_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, VECTOR_DIM))

    test_coarse_old_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    test_coarse_new_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    test_global_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))
    test_truth_node = tf.placeholder(tf.float32,
        shape=(1, GRID_SIZE, GRID_SIZE, VECTOR_DIM))

    node, alpha = model(coarse_old_node, coarse_new_node, global_node)
    predict, _ = model(test_coarse_old_node, test_coarse_new_node, test_global_node)

    loss = loss_function(node, truth_node)

    learning_rate = 0.00001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print "optimizer:",optimizer
    with tf.Session() as s:
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        # get data
        coarse_old_data, coarse_new_data, global_data, truth_data = load_all_data(0)
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

            _, l, a = s.run(
                [optimizer, loss, alpha],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %.1f, step %d, progress %.2f%% loss: %.4f, alpha %.4f'\
                    % (float(step) * BATCH_SIZE / DATA_SIZE, step, 100.*step/total_step, l, a)
                sys.stdout.flush()

            # save the model
            if int(step+1) % 6000 == 0:
                saver = tf.train.Saver()
                path = saver.save(s, './save/cnn_test_model')
                print 'Saving result to ' + path

if __name__ == '__main__':
    tf.app.run()
