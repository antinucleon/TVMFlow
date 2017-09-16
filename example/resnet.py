import tvmflow as tf
import numpy as np

num_stages = 4
filter_list = [64, 64, 128, 256, 512]
units = [2, 2, 2, 2]
bottle_neck = False


def res_unit(data, out_filter, in_filter, stride, dim_match, name):
    bn1 = tf.nn.batch_norm(data, in_filter, name=name + "_bn1")
    act1 = tf.nn.relu(bn1, name=name + "_relu1")
    conv1 = tf.nn.conv2d(
        act1,
        out_filter,
        in_filter,
        kernel=3,
        stride=stride,
        pad="SAME",
        name=name + "_conv1")
    bn2 = tf.nn.batch_norm(conv1, out_filter, name=name + "_bn2")
    act2 = tf.nn.relu(bn2, name=name + "_relu2")
    conv2 = tf.nn.conv2d(
        act2,
        out_filter,
        out_filter,
        kernel=3,
        stride=1,
        pad="SAME",
        name=name + "_conv2")
    if dim_match:
        short_cut = data
    else:
        short_cut = tf.nn.conv2d(
            data,
            out_filter,
            in_filter,
            kernel=1,
            stride=stride,
            pad="SAME",
            name=name + "_sc")

    return conv2 + short_cut


batch = tf.placeholder(tf.float32, [None, 3, 224, 224], name="img")
#data = tf.nn.batch_norm(batch, 3, name="norm")

# stage 0
body = tf.nn.conv2d(
    batch,
    out_filter=filter_list[0],
    in_filter=3,
    kernel=7,
    stride=2,
    pad="SAME",
    name="conv0")
"""
body = tf.nn.batch_norm(body, 64, name="bn0")

#body = tf.nn.relu(body, name="relu0")

#body = tf.nn.max_pool(net, stride=2)

body = tf.nn.conv2d(
    body,
    out_filter=filter_list[0],
    in_filter=filter_list[0],
    kernel=3,
    stride=2,
    pad="SAME",
    name="conv1")

for i in range(num_stages):
    body = res_unit(
        body,
        filter_list[i + 1],
        filter_list[i],
        1 if i == 0 else 2,
        False,
        name='stage%d_unit%d' % (i + 1, 1))

    for j in range(units[i] - 1):
        body = res_unit(
            body,
            filter_list[i + 1],
            filter_list[i + 1],
            1,
            True,
            name='stage%d_unit%d' % (i + 1, j + 2))

bn1 = tf.nn.batch_norm(body, filter_list[-1], name="bn1")
relu1 = tf.nn.relu(bn1, name="relu1")
pool1 = tf.nn.global_pool(relu1, name="gp")
flat = tf.nn.flatten(pool1, name="flatten")

fc1 = tf.nn.fully_connected(flat, 1000, filter_list[-1], name="fc1")
net = tf.nn.softmax(fc1, name="softmax")

gx = tf.gradients(net, [batch])
"""
sess = tf.Session("metal, float32")
sess.run(tf.initialize_all_variables())
print("-" * 60)
print sess.run(body, feed_dict={batch: np.ones((2, 3, 224, 224)) / 256.}).shape
