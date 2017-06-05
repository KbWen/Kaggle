import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# read training data from CSV file
train_data = pd.read_csv('train.csv')
# print(train_data.columns)
# print(train_data.head(1))
# print(train_data.describe())
# print(train_data.index)
images = train_data.iloc[:,1:].values.astype(np.float)
# [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
# print(images[0])
# print('data shape:{0}'.format(train_data.shape))
# print('images shape:{0}'.format(images.shape))
# print ('image_size => {0}'.format(image_size))
image_width = image_height = np.sqrt(image_size).astype(np.uint8)

# display image
def display(img):
    one_image = img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    # plt.show()
display(images[992])

# Return a flattened array.
train_labels_flat = train_data[[0]].values.ravel()
# Find the unique elements of an array
train_labels_count = np.unique(train_labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    print(labels_one_hot)
    return labels_one_hot

labels = dense_to_one_hot(train_labels_flat, train_labels_count)
labels = labels.astype(np.uint8)

# print('labels({0[0]},{0[1]})'.format(labels.shape))
# print ('labels[{0}] => {1}'.format(0,labels[0]))

# train data & validation data
validation_images = images[:2000]
validation_labels = labels[:2000]
train_images = images[2000:]
train_labels = labels[2000:]

x_ = tf.placeholder(tf.float32, [None,image_size])  # 28*28
y_ = tf.placeholder(tf.float32, [None,train_labels_count])
x_image = tf.reshape(x_, [-1,image_width,image_height,1])  # Gray scale:1  RBG:3
keep_prob = tf.placeholder(tf.float32)
# define W, b convolution
def Weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.12, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

# con1 use 5*5  1 to 32
W_conv1 = Weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
hidden_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
hidden_pool1 = max_pool_2x2(hidden_conv1)
# con2
W_conv2 = Weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
hidden_conv2 = tf.nn.relu(conv2d(hidden_pool1, W_conv2) + b_conv2)
hidden_pool2 = max_pool_2x2(hidden_conv2)

# hidden layer 1   28*28 -- 14*14 -- 7*7
hidden_x1 = tf.reshape(hidden_pool2, [-1, 7*7*64])
hidden_W1 = Weight_variable([7*7*64, 512])
hidden_b1 = bias_variable([512])
hidden_act1 = tf.nn.relu(tf.matmul(hidden_x1, hidden_W1) + hidden_b1)
# dropout
hidden_drop = tf.nn.dropout(hidden_act1, keep_prob)
# output layer
prediction_W = Weight_variable([512,10])
prediction_b = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(hidden_drop, prediction_W) + prediction_b)
# [True, False, False, False, False] = [1, 0, 0, 0, 0] = 0.2
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_prediction = sess.run(prediction, {x_:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # result  = sess.run(accuracy, {x_: v_xs, y_:v_ys keep_prob:1})
    result = accuracy.eval({x_: v_xs, y_:v_ys, keep_prob:1})
    return result
# cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction),
                               reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.06).minimize(cross_entropy)
# initial
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
##sess.run(tf.global_variables_initializer())

# train batch
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += 50

    if index_in_epoch > num_examples:
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = 50
        assert 50 <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


for i in range(2500):
    batch = next_batch(50) ##Stochastic gradient descent
    sess.run(train_step, feed_dict={x_:batch[0], y_:batch[1], keep_prob:0.6})
    if i %50==0:
        print(compute_accuracy(validation_images,validation_labels))

# test CSV
predict = tf.argmax(prediction,1)
test_images = pd.read_csv('test.csv').values.astype(np.float)
test_images = np.multiply(test_images, 1.0/255.0)
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//50):
    predicted_lables[i*50 : (i+1)*50] = predict.eval(
    feed_dict={x_: test_images[i*50 : (i+1)*50], keep_prob: 1.0})

# save results
np.savetxt('submission_test.csv',
           np.c_[range(1,len(test_images)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')
