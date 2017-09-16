"""Tinyflow example code.

Minimum softmax code that exposes the optimizer.
"""
import tvmflow as tf
import numpy as np
from tvmflow.datasets import get_mnist

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name="x")
W = tf.Variable(tf.zeros(shape=[784, 10], dtype=tf.float32))
h = tf.matmul(x, W)
y = tf.softmax(h)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10], name="label")
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), keepdims=True)

learning_rate = 5e-1

W_grad = tf.gradients(cross_entropy, [W])[0]
train_step = tf.assign(W, W - learning_rate * W_grad)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

# get the mnist dataset
mnist = get_mnist(flatten=True, onehot=True)

print("start training")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    out = tf.Group([train_step, cross_entropy])
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #print ce
    #raw_input()
# sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(correct_prediction, keepdims=True)

print(sess.run(
    accuracy, feed_dict={x: mnist.test.images,
                         y_: mnist.test.labels}))
