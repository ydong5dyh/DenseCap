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

def rnn_network(input_image_feature, input_data, keep_prob, model='lstm', rnn_size=256, num_layers=1, reuse=None):
    len_words = 3000
    image_feat_size = 2048

    if model == 'rnn':
        cell_fun = rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(1, tf.float32)

    with tf.variable_scope('rnnlm', reuse=reuse):
        with tf.device("/gpu:0"):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, len_words])
            softmax_b = tf.get_variable("softmax_b", [len_words])
            Wi = tf.get_variable("Wi", [image_feat_size, rnn_size])
            bi = tf.get_variable("bi", [rnn_size])

            embedding = tf.get_variable("embedding", [len_words, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

            inputs = tf.nn.dropout(inputs, keep_prob)

            image_med = tf.matmul(input_image_feature, Wi) + bi

            image_med = tf.nn.dropout(image_med, keep_prob)

            inputs = tf.reshape(inputs, [-1, rnn_size])
            input_stack = tf.concat([image_med, inputs], 0)
            input_stack = tf.reshape(input_stack, [1, -1, rnn_size])
            image_med = tf.reshape(image_med, [1, -1, rnn_size])
            inputs = tf.reshape(inputs, [1, -1, rnn_size])

            outputs, last_state = tf.nn.dynamic_rnn(cell, input_stack, initial_state=initial_state, scope='rnnlm')
            output = tf.reshape(outputs, [-1, rnn_size])
            img_outs, img_last_state = tf.nn.dynamic_rnn(cell, image_med, initial_state=initial_state, scope='rnnlm')
            img_out = tf.reshape(img_outs, [-1, rnn_size])
            word_outs, word_last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
            word_out = tf.reshape(word_outs, [-1, rnn_size])

            output = tf.nn.dropout(output, keep_prob)
            img_out = tf.nn.dropout(img_out, keep_prob)
            word_out = tf.nn.dropout(word_out, keep_prob)

            logits = tf.matmul(output, softmax_w) + softmax_b
            probs = tf.nn.softmax(logits)
            img_logits = tf.matmul(img_out, softmax_w) + softmax_b
            img_probs = tf.nn.softmax(img_logits)
            word_logits = tf.matmul(word_out, softmax_w) + softmax_b
            word_probs = tf.nn.softmax(word_logits)
    return logits, last_state, probs, cell, initial_state, img_probs, img_last_state, word_probs, word_last_state

caffe.set_mode_gpu()
model_def = './VGG-16/deploy.prototxt'
model_weights = './VGG-16/VGG_ILSVRC_16_layers.caffemodel'
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

word2idx = cPickle.load(open("word-index/word2idx.pkl", "rb"))
idx2word = cPickle.load(open("word-index/idx2word.pkl", "rb"))

rep_size = 256
len_words = 3000
feat_size = 2048
keep = 0.5
is_train = 0
total_train = []
total_regions = []
total_feature_map = []
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

input_feature = tf.placeholder(tf.float32, [1, feat_size])
input_data = tf.placeholder(tf.int64, [1, None])
output_label = tf.placeholder(tf.int64, [1, None])
keep_prob = tf.placeholder(tf.float32)
feat = tf.placeholder(tf.float32, [])

feat_medut = tf.placeholder(tf.float32, [None, conv_height, conv_width, 512])
with tf.variable_scope('rcnn', reuse=None):
    W_conv6 = tf.Variable(tf.truncated_normal([3, 3, 512, 256], mean=0, stddev=0.1), name="W_conv6")
    b_conv6 = tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.1), name="b_conv6")
    feat = tf.nn.conv2d(feat_medut, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6

    W_offset = tf.Variable(tf.truncated_normal([1, 1, 256, k * 4], mean=0, stddev=0.1), name="W_offset")
    b_offset = tf.Variable(tf.truncated_normal([k * 4], mean=0, stddev=0.1), name="b_offset")
    offset = tf.nn.conv2d(feat, W_offset, strides=[1, 1, 1, 1], padding='SAME') + b_offset
    offset = tf.reshape(offset, [k * conv_height * conv_width, 4])

    W_score =  tf.Variable(tf.truncated_normal([1, 1, 256, k], mean=0, stddev=0.1), name="W_score")
    b_score = tf.Variable(tf.truncated_normal([k], mean=0, stddev=0.1), name="b_score")
    score =  tf.nn.conv2d(feat, W_score, strides=[1, 1, 1, 1], padding='SAME') + b_score
    score = tf.reshape(score, [k * conv_height * conv_width])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, 'RPN_model/fasterRcnn.module-10')

description = json.load(open("./VG/region_descriptions.json", "rb"))
for img_id in range(1, 11):

    img = PIL_Image.open("./VG/VG_100K/" + str(img_id) + ".jpg")
    feature = img2feature(img, net)
    feature = np.transpose(feature, [0, 2, 3, 1])

    ofs = sess.run(offset, feed_dict={feat_medut: feature})
    scr = sess.run(score, feed_dict={feat_medut: feature})

    regions = description[img_id - 1]["regions"]
    size = img.size
    origin_width = size[0]
    origin_height = size[1]
    w_scale = width / float(origin_width)
    h_scale = height / float(origin_height)
    ground_truth_ = []
    for idx in range(ground_truth_num):
        rgt = [int(round(regions[idx]["y"] * h_scale)),
               int(round(regions[idx]["x"] * w_scale)),
               int(round(regions[idx]["height"] * h_scale + (h_scale - 1))),
               int(round(regions[idx]["width"] * w_scale + (w_scale - 1)))]
        ground_truth_.append(rgt)
    ground_truth_ = np.array((ground_truth_))

    sco = scr.reshape(14, 14, k).transpose(2, 0, 1)
    result = ofs.reshape(14, 14, 4 * k).transpose(2, 0, 1)
    score_index = np.array((np.where(sco > 0.5)))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(img)
    infer = []
    iscore = []
    for i in range(score_index.shape[1]):
        bounding_index = i
        bounding_k = score_index[0, bounding_index]
        bounding_y = score_index[1, bounding_index]
        bounding_x = score_index[2, bounding_index]
        Y = (bounding_y * float(600)) / 13
        X = (bounding_x * float(800)) / 13

        kth = bounding_k
        (h, w) = bbox[kth]
        pos_infer = result[bounding_k * 4:bounding_k * 4 + 4, bounding_y, bounding_x]
        y = Y + pos_infer[0] * h
        x = X + pos_infer[1] * w
        h = h * np.exp(pos_infer[2])
        w = w * np.exp(pos_infer[3])
        y = y - h / 2
        x = x - w / 2
        if x < 0 or y < 0 or h < 5 or w < 5 or x + w > 800 or y + h > 600:
            continue
        infer.append([y, x, h, w])
        iscore.append([sco[bounding_k, bounding_y, bounding_x]])

    infer = np.array(infer).reshape(-1, 4)
    iscore = np.array(iscore).reshape(1, -1)
    num = infer.shape[0]
    infer = tf.cast(infer, tf.float32)
    iscore = tf.cast(iscore, tf.float32)
    nms_infer, nms_score = nms(infer, iscore, num)
    nms_infer = sess.run(nms_infer)
    nms_score = sess.run(nms_score)

    num = nms_infer.shape[0]
    nms_infer = tf.cast(nms_infer, tf.float32)
    ground_truth_ = tf.cast(ground_truth_, tf.float32)
    nms_iou = get_iou(ground_truth_, 10, nms_infer, num)
    ground_truth_ = tf.argmax(nms_iou, axis=1)

    nms_infer = sess.run(nms_infer)
    ground_truth_ = sess.run(ground_truth_)
    ground_truth_ = sess.run(ground_truth_)

    nms_infer = nms_infer.reshape(-1, 4)
    ground_truth_ = ground_truth_.reshape(-1, 1)
    train_data = np.concatenate([nms_infer, ground_truth_], axis=1)
    feature_map = img2feature(img, net)
    feature_map = feature_map.reshape(512, 14, 14)
    feature_map = feature_map.transpose(1, 2, 0)

    feature_map = []
    for i in range(train_data.shape[0]):
        Y, X, H, W = train_data[i, :4]
        w_scale = float(14) / width
        h_scale = float(14) / height
        y = int(round(Y * h_scale))
        x = int(round(X * w_scale))
        h = int(round(H * h_scale + (h_scale - 1)))
        w = int(round(W * w_scale + (w_scale - 1)))
        sfeature_map = feature_map[y:y + h + 1, x:x + w + 1, :]
        input_y = sfeature_map.shape[0]
        input_x = sfeature_map.shape[1]
        sfeature_map = sfeature_map.reshape(1, input_y, input_x, 512)
        sfeature_map = tf.cast(sfeature_map, tf.float32)
        if input_y >= 2 and input_x >= 2:
            sfeature_map = tf.nn.max_pool(sfeature_map, ksize=[1, int(round(float(input_y) / 2)), int(round(float(input_x) / 2)), 1],
                                     strides=[1, input_y / 2, input_x / 2, 1], padding='VALID')
        elif input_y >= 2:
            sfeature_map = tf.nn.max_pool(sfeature_map, ksize=[1, int(round(float(input_y) / 2)), 1, 1],
                                     strides=[1, input_y / 2, 1, 1], padding='VALID')
            sfeature_map = tf.tile(sfeature_map, [1, 1, 2, 1])
        elif input_x >= 2:
            sfeature_map = tf.nn.max_pool(sfeature_map, ksize=[1, 1, int(round(float(input_x) / 2)), 1],
                                     strides=[1, 1, input_x / 2, 1], padding='VALID')
            sfeature_map = tf.tile(sfeature_map, [1, 2, 1, 1])
        else:
            sfeature_map = tf.tile(sfeature_map, [1, 2, 2, 1])
        sfeature_map = tf.reshape(sfeature_map, [1, 2048])
        sfeature_map = sess.run(sfeature_map)
        feature_map.append(sfeature_map)
    feature_map = np.concatenate(feature_map, axis=0)
    total_train.append(train_data)
    total_regions.append(regions)
    total_feature_map.append(feature_map)
sess.close()


print("Training Started!")
logits, last_state, _, _, _, _, _, _, _ = rnn_network(input_feature, input_data, keep_prob)
labels = tf.reshape(output_label, [-1])
loss = seq2seq.sequence_loss_by_example([logits], [labels], [tf.ones_like(labels, dtype=tf.float32)], len_words)
cost = tf.reduce_mean(loss)
learning_rate = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    for epoch in range(500):
        sess.run(tf.assign(learning_rate, 0.001 * (0.9 ** (epoch / 100))))
        for k in range(len(total_train)):
            train_data = total_train[k]
            regions = total_regions[k]
            feature_map = total_feature_map[k]
            for i in range(train_data.shape[0]):
                train, test = sen2ix(regions[train_data[i, 4].astype('int32')]['phrase'], wordtoix)
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: train,
                                                       input_image_feature: feature_map[i].reshape(1, 2048),
                                                       output_labels: test, keep_prob: keep})
    if epoch % 50 == 0:
        print(epoch, train_loss)
        saver.save(sess, 'RNN_model/test.module')
print("Training Finished!")
