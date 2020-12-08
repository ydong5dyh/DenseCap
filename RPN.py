import caffe
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import time
import io
import cPickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image as PIL_Image
import requests
from StringIO import StringIO
import json

def img2feature(img, net):
    img = np.array(img.resize([224, 224]))
    net.blobs['data'].data[...] = img.transpose([2, 0, 1])
    net.forward()
    feat = net.blobs['conv5_3'].data
    return feat

def generate_anchors(boxes, height, width, conv_height, conv_width):
    k, _ = boxes.get_shape().as_list()

    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    grid = tf.transpose(tf.stack(tf.meshgrid(
        tf.linspace(0.0, height - 0.0, conv_height),
        tf.linspace(0.0, width - 0.0, conv_width)), axis=2), [1, 0, 2])

    # convert boxes from K x 2 to 1 x 1 x K x 2
    boxes = tf.expand_dims(tf.expand_dims(boxes, 0), 0)
    # convert grid from H' x W' x 2 to H' x W' x 1 x 2
    grid = tf.expand_dims(grid, 2)

    # combine them into single H' x W' x K x 4 tensor
    return tf.concat([tf.tile(grid, [1, 1, k, 1]),
                      tf.tile(boxes, [conv_height, conv_width, 1, 1])], 3)

def get_iou(ground_truth, ground_truth_count, proposals, proposals_count):
    proposals = tf.expand_dims(proposals, axis=1)
    proposals = tf.tile(proposals, [1, ground_truth_count, 1])

    ground_truth = tf.expand_dims(ground_truth, axis=0)
    ground_truth = tf.tile(ground_truth, [proposals_count, 1, 1])

    ycord_11, xcord_11, height1, width1 = tf.unstack(proposals, axis=2)
    ycord_21, xcord_21, height2, width2 = tf.unstack(ground_truth, axis=2)

    x11, y11 = xcord_11 - width1 // 2, ycord_11 - height1 // 2
    x21, y21 = xcord_21 - width2 // 2, ycord_21 - height2 // 2
    x12, y12 = x11 + width1, y11 + height1
    x22, y22 = x21 + width2, y21 + height2

    intersection = (
        tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21)) *
        tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))
    )

    iou = intersection / (
        width1 * height1 + width2 * height2 - intersection
    )
    return iou

caffe.set_mode_gpu()
model_def = './VGG-16/deploy.prototxt'
model_weights = './VGG-16/VGG_ILSVRC_16_layers.caffemodel'
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

boxes = tf.Variable([
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
], dtype=tf.float32)
bbox = [
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512)
]

conv_height = 14
conv_width = 14
height = 600
width = 800
k = 12
ground_truth_num = 10
anchors_num = k * conv_height * conv_width
img_num = 10
img_Epoch = 200000
train_img = np.random.permutation(img_num) + 1
description = json.load(open("./VG/region_descriptions.json", "rb"))

feat_input = tf.placeholder(tf.float32, [None, conv_height, conv_width, 512])

with tf.variable_scope('rcnn', reuse=None):
    W_conv6 = tf.Variable(tf.truncated_normal([3, 3, 512, 256], mean=0, stddev=0.1), name="W_conv6")
    b_conv6 = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1), name="b_conv6")
    feat = tf.nn.conv2d(feat_input, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6

    W_offset = tf.Variable(tf.truncated_normal([1, 1, 256, k * 4], mean=0, stddev=0.1), name="W_offset")
    b_offset = tf.Variable(tf.truncated_normal([k * 4], mean=0, stddev=0.1), name="b_offset")
    offset = tf.nn.conv2d(feat, W_offset, strides=[1, 1, 1, 1], padding='SAME') + b_offset
    offset = tf.reshape(offset, [k * conv_height * conv_width, 4])

    W_score =  tf.Variable(tf.truncated_normal([1, 1, 256, k], mean=0, stddev=0.1), name="W_score")
    b_score = tf.Variable(tf.truncated_normal([k], mean=0, stddev=0.1), name="b_score")
    score =  tf.nn.conv2d(feat, W_score, strides=[1, 1, 1, 1], padding='SAME') + b_score
    score = tf.reshape(score, [k * conv_height * conv_width])

anchors = generate_anchors(boxes, height, width, conv_height, conv_width)
anchors = tf.reshape(anchors, [-1, 4])
ground_truth_pre = tf.placeholder(tf.float32, [None, 4])
y, x, height, width = tf.unstack(ground_truth_pre, axis=1)
yc, xc = y + height // 2, x + width // 2
ground_truth = tf.stack([yc, xc, height, width], axis=1)
iou = get_iou(ground_truth, ground_truth_num, anchors, anchors_num)
positive_mask = tf.reduce_any(tf.greater_equal(iou, 0.5), axis=1)

    # Sample would be considered negative if _all_ ground truch box
    # have iou less than 0.3
negative_mask = tf.reduce_all(tf.less(iou, 0.3), axis=1)

    # Select only positive boxes and their corresponding predicted scores
positive_boxes = tf.boolean_mask(proposals, positive_mask)
positive_scores = tf.boolean_mask(scores, positive_mask)
positive_labels = tf.ones_like(positive_scores)

    # Same for negative
negative_boxes = tf.boolean_mask(proposals, negative_mask)
negative_scores = tf.boolean_mask(scores, negative_mask)
negative_labels = tf.zeros_like(negative_scores)

predicted_scores = tf.concat([positive_scores, negative_scores], 0)
true_labels = tf.concat([positive_labels, negative_labels], 0)
score_loss = tf.reduce_sum(tf.square(predicted_scores - true_labels))

ground_truth = tf.expand_dims(ground_truth, axis=0)
ground_truth = tf.tile(ground_truth, [anchors_num, 1, 1])
# anchor_centers shape is N x 4 where N is count and 4 are ya,xa,ha,wa
anchors = tf.expand_dims(anchors, axis=1)
anchors = tf.tile(anchors, [1, ground_truth_num, 1])
y_anchor, x_anchor, height_anchor, width_anchor = tf.unstack(anchors, axis=2)
y_ground_truth, x_ground_truth, height_ground_truth, width_ground_truth = tf.unstack(ground_truth, axis=2)

tx_ground_truth = (x_ground_truth - x_anchor) / width_anchor
ty_ground_truth = (y_ground_truth - y_anchor) / height_anchor
tw_ground_truth = tf.log(width_ground_truth / width_anchor)
th_ground_truth = tf.log(height_ground_truth / height_anchor)

ground_truth_param = tf.stack([ty_ground_truth, tx_ground_truth, th_ground_truth, tw_ground_truth], axis=2)

pos_mask = tf.reduce_any(tf.greater_equal(iou, 0.5), axis=1)
pos_gt_params = tf.boolean_mask(ground_truth_param, pos_mask)
pos_offset = tf.boolean_mask(offset, pos_mask)
pos_iou = tf.boolean_mask(iou, pos_mask)
max_mask = tf.equal(tf.one_hot(tf.argmax(pos_iou, axis=1), ground_truth_num, 1, 0, -1), 1)
pos_offset_labels = tf.reshape(tf.boolean_mask(pos_gt_params, max_mask), [-1, 4])
pos_offset, pos_offset_labels = get_offset_labels(ground_truth_param, ground_truth_num, offset, iou)
offset_loss = tf.reduce_sum(tf.square(pos_offset - pos_offset_labels))

total_loss = score_loss + offset_loss
learning_rate = tf.Variable(0.0, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
print "start training ..."
for img_epoch in range(img_Epoch):
    sess.run(tf.assign(learning_rate, 0.0001 * (0.98 ** (img_epoch / 10000))))
    for img_id in train_img:
        img = PIL_Image.open("./VG/VG_100K/" + str(img_id) + ".jpg")
        regions = description[img_id - 1]["regions"]
        size = img.size
        origin_width = size[0]
        origin_height = size[1]
        w_scale = width / float(origin_width)
        h_scale = height / float(origin_height)
        ground_truth_ = []
        for idx in range(ground_truth_num):
            rground_truth = [int(round(regions[idx]["y"] * h_scale)),
                   int(round(regions[idx]["x"] * w_scale)),
                   int(round(regions[idx]["height"] * h_scale + (h_scale - 1))),
                   int(round(regions[idx]["width"] * w_scale + (w_scale - 1)))]
            ground_truth_.append(rground_truth)
        ground_truth_ = np.array((ground_truth_))
        feature = img2feature(img, net)
        feature = np.transpose(feature, [0, 2, 3, 1])
        Epoch = 7
        with tf.device("/gpu:0"):
            for epoch in range(Epoch):
                sess.run([train_step], feed_dict={feat_input: feature, ground_truth_pre: ground_truth_})
    if img_epoch % (img_Epoch / 100) == 0:
        print "epoch:", img_epoch, "img:", img_id, sess.run([total_loss],
                                                            feed_dict={feat_input: feature,
                                                                       ground_truth_pre: ground_truth_})
    if (img_epoch + 1) % (img_Epoch / 10) == 0:
        saver.save(sess, 'RPN_model/fasterRcnn.module', global_step=img_epoch + 1)
sess.close()
print "train end"
