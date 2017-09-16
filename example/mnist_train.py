"""Tinyflow example code.

This code is adapted from Tensorflow's MNIST Tutorial with minimum code changes.
"""
import tvmflow as tf
from tvmflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name="x")

fc1 = tf.nn.fully_connected(x, 32, 784, "fc1")
act1 = tf.nn.relu(fc1, "relu1")
fc2 = tf.nn.fully_connected(x, 10, 784, "fc2")
y = tf.nn.softmax(fc2, name="sm")

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

loss = tf.nn.logloss(y, y_)

train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    out = tf.Group([train_step, loss])
    ce = sess.run(out, feed_dict={x: batch_xs, y_: batch_ys})[-1]
    if i % 100 == 0:
        print ce

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(correct_prediction, keepdims=True)

print(sess.run(
    accuracy, feed_dict={x: mnist.test.images,
                         y_: mnist.test.labels}))
