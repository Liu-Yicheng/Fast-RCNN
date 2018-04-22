import tensorflow as tf
from tensorflow.python.ops import nn_ops
import config as cfg
from roi_pooling.roi_pooling_ops import roi_pooling
slim = tf.contrib.slim

class Alexnet(object):

    def __init__(self,  is_training=True):
        self.is_training = is_training
        self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        self.class_num = cfg.Class_num
        self.image_w = cfg.Image_w
        self.image_h = cfg.Image_h
        self.batch_size = cfg.Batch_size
        if is_training:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_h, self.image_w, 3], name='input')
        else:
            self.images = tf.placeholder(tf.float32, [1, self.image_h, self.image_w, 3], name='test_input')
        self.rois = tf.placeholder(tf.int32,[None, 5], name='rois')
        self.logits, self.bbox = self.build_network(self.images, self.class_num, self.is_training)
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.class_num*5-4 ], name='labels')
            self.loss_layer(self.logits, self.labels, self.bbox)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total-loss', self.total_loss)

    def build_network(self, images, class_num, is_training=True, keep_prob=0.5, scope='Fast-RCNN'):

        self.conv1 = self.convLayer(images, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = self.LRN(self.conv1, 2, 2e-05, 0.75, "norm1")
        self.pool1 = self.maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")
        self.conv2 = self.convLayer(self.pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = self.LRN(self.conv2, 2, 2e-05, 0.75, "lrn2")
        self.pool2 = self.maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")
        self.conv3 = self.convLayer(self.pool2, 3, 3, 1, 1, 384, "conv3")
        self.conv4 = self.convLayer(self.conv3, 3, 3, 1, 1, 384, "conv4", groups=2)
        self.conv5 = self.convLayer(self.conv4, 3, 3, 1, 1, 256, "conv5", groups=2)

        self.roi_pool6 = roi_pooling(self.conv5, self.rois, pool_height=6, pool_width=6)

        with slim.arg_scope([slim.fully_connected, slim.conv2d],
                            activation_fn=nn_ops.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            flatten = slim.flatten(self.roi_pool6, scope='flat_32')
            self.fc1 = slim.fully_connected(flatten, 4096,  scope='fc_6')
            drop6 = slim.dropout(self.fc1, keep_prob=keep_prob, is_training=is_training, scope='dropout6',)
            self.fc2 = slim.fully_connected(drop6,  4096,  scope='fc_7')
            drop7 = slim.dropout(self.fc2, keep_prob=keep_prob, is_training=is_training, scope='dropout7')
            cls = slim.fully_connected(drop7, class_num,activation_fn=nn_ops.softmax ,scope='fc_8')
            bbox = slim.fully_connected(drop7, (self.class_num-1)*4,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.001),
                                        activation_fn=None ,scope='fc_9')
        return cls,bbox

    def maxPoolLayer(self, x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
        return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)

    def LRN(self, x, R, alpha, beta, name=None, bias=1.0):
        return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha,
                                              beta=beta, bias=bias, name=name)

    def convLayer(self, x, kHeight, kWidth, strideX, strideY,
                  featureNum, name, padding="SAME", groups=1):  # group为2时等于AlexNet中分上下两部分
        channel = int(x.get_shape()[-1])  # 获取channel
        conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)  # 定义卷积的匿名函数
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
            b = tf.get_variable("b", shape=[featureNum])
            tf.losses.add_loss(self.regularizer(w))
            xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)  # 划分后的输入和权重
            wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

            featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]  # 分别提取feature map
            mergeFeatureMap = tf.concat(axis=3, values=featureMap)  # feature map整合
            out = tf.nn.bias_add(mergeFeatureMap, b)
            return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()))  # relu后的结果

    def loss_layer(self, y_pred, y_true, box_pred):
        cls_pred = y_pred
        cls_true = y_true[:, :self.class_num]
        bbox_pred = box_pred
        bbox_ture = y_true[:, self.class_num:]

        cls_pred /= tf.reduce_sum(cls_pred,
                                 reduction_indices=len(cls_pred.get_shape()) - 1,
                                  keep_dims=True)
        cls_pred = tf.clip_by_value(cls_pred, tf.cast(1e-10, dtype=tf.float32), tf.cast(1. - 1e-10, dtype=tf.float32))
        cross_entropy = -tf.reduce_sum(cls_true * tf.log(cls_pred), reduction_indices=len(cls_pred.get_shape()) - 1)
        cls_loss = tf.reduce_mean(cross_entropy)
        tf.losses.add_loss(cls_loss)
        tf.summary.scalar('class-loss', cls_loss)

        mask = tf.tile(tf.reshape(cls_true[:, 1], [-1, 1]), [1, 4])
        for cls_idx in range(2, self.class_num):
            mask =tf.concat([mask, tf.tile(tf.reshape(cls_true[:, int(cls_idx)], [-1, 1]), [1, 4])], 1)
        bbox_sub =  tf.square(mask * (bbox_pred - bbox_ture))
        bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_sub, 1))
        tf.losses.add_loss(bbox_loss)
        tf.summary.scalar('bbox-loss', bbox_loss)



