import tvmflow as tf
import numpy as np


def test_add_grad():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    ay = np.ones((2, 3)) * 4
    z = x + y
    gx, gy = tf.gradients(z, [x, y])
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax, y: ay})
    np.testing.assert_almost_equal(agx, np.ones((2, 3)))


def test_mul_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = x * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax})
    np.testing.assert_almost_equal(agx, np.ones((2, 3)) * 14)


def test_sum_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = -tf.reduce_sum(x, reduction_indices=[1]) * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax})
    np.testing.assert_almost_equal(agx, -np.ones((2, 3)) * 14)


def test_mean_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    z = -tf.reduce_mean(x, reduction_indices=[1]) * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax})
    np.testing.assert_almost_equal(agx, -np.ones((2, 3)) * 14 / 3.0, decimal=4)


def test_mean2_grad():
    x = tf.placeholder(tf.float32)
    ax = np.ones((7))
    z = -tf.reduce_mean(x, keepdims=True) * 14
    gx = tf.gradients(z, [x])[0]
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax})
    np.testing.assert_almost_equal(agx, -np.ones((7)) * 14 / 7.0, decimal=4)

def test_matmul_grad():
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    ax = np.ones((2, 3))
    ay = np.ones((3, 4)) * 4
    z = tf.matmul(x, y) * 4
    gx, gy = tf.gradients(z, [x, y])
    sess = tf.Session("cpu, float32")
    agx = sess.run(gx, feed_dict={x: ax, y: ay})
    agy = sess.run(gy, feed_dict={x: ax, y: ay})
    np.testing.assert_almost_equal(
        agx,
        np.dot(np.ones((2, 4)), ay.T) * 4)
    np.testing.assert_almost_equal(
        agy,
        np.dot(ax.T, np.ones((2, 4))) * 4)


if __name__ == "__main__":
    test_add_grad()
    test_mul_grad()
    test_sum_grad()
    test_mean_grad()
    test_mean2_grad()
    test_matmul_grad()
    # pass
