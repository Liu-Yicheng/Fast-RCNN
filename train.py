import os
import Net
import numpy as np
import Data
import config as cfg
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib
matplotlib.use('TkAgg')
class Solver(object):
    def __init__(self, net, data, is_training=True):
        self.net = net
        self.data = data

        self.max_iter = cfg.Max_iter
        self.save_iter = cfg.Save_iter
        self.summary_iter = cfg.Summary_iter
        self.output_dir = './Output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.weights_file = cfg.Weights_file
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)
        exclude = ['fc_6/weights', 'fc_7/weights', 'fc_8/weights', 'fc_9/weights',
                   'fc_6/biases', 'fc_7/biases', 'fc_8/biases',
                   'fc_9/biases', 'is_training']
        self.variable_to_restore = slim.get_variables_to_restore(exclude=exclude)
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=1)
        self.saver = tf.train.Saver(max_to_keep=1)
        if is_training:
            self.initial_learning_rate = cfg.Initial_learning_rate
            self.decay_rate = cfg.Decay_rate
            self.decay_iter = cfg.Decay_iter
            self.staircase = cfg.Staircase
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                                                            self.initial_learning_rate,
                                                            self.global_step,
                                                            self.decay_iter,
                                                            self.decay_rate,
                                                            self.staircase,
                                                            name='learning_rate'
                                                            )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                                                    self.net.total_loss, global_step=self.global_step)
            self.ema = tf.train.ExponentialMovingAverage(0.9)
            self.average_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.average_op)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            self.restorer.restore(self.sess, self.weights_file)
	    print('loading the model weight...')
        self.writer.add_graph(self.sess.graph)
        self.save_cfg()

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def train(self):
        for step in range(1, self.max_iter + 1):
            images, rois, labels = self.data.get_batch()
            feed_dict = {self.net.images: images, self.net.rois: rois, self.net.labels: labels}
            if step % self.summary_iter == 0:

                summary, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.train_op],
                                                 feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
                print("Data_epoch:" + str(self.data.epoch) + " " * 5 + "training_step:" + str(
                    step) + " " * 5 + "batch_loss:" + str(loss))

            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)
            if step % self.save_iter == 0:
                print("saving the model as : " + self.ckpt_file)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)
                

    def predict(self):
        class_num = cfg.Class_num
        image_path = cfg.Images_path
        with open(r'./Data/test_list.txt','r') as f:
            test_index_collect = f.readlines()
        for test_index in test_index_collect:
            test_index = test_index.strip()
            regions = []
            results_dic = {}
            for cls_index in range(1, class_num+1):
                results_dic[cls_index] = []
            images, roises, labels = self.data.get_valid_batch(test_index)
            feed_dict = {self.net.images: images, self.net.rois: roises}
            results, bbox_ = self.sess.run([self.net.logits, self.net.bbox], feed_dict=feed_dict)
            print('********************************************')
            for index in range(len(results)):
                if (np.argmax(results[index][0:class_num])) != 0 and np.max(results[index][0:class_num] > 0.5):
                    rois = roises[index][1:5] * 16
                    regions_old = [(rois[0] + rois[2]) / 2.0, (rois[1] + rois[3]) / 2.0, rois[2] - rois[0],
                                   rois[3] - rois[1]]
                    ind = np.argmax(results[index]) - 1
                    score = np.max(results[index][0:5])
                    x_rate, y_rate, w_rate, h_rate = bbox_[index][ind * 4], bbox_[index][ind * 4 + 1], \
                                                     bbox_[index][ind * 4 + 2], bbox_[index][ind * 4 + 3]

                    region_new = [regions_old[0] + regions_old[2] * x_rate, regions_old[1] + regions_old[3] * y_rate,
                                  regions_old[2] * np.exp(w_rate), regions_old[3] * np.exp(h_rate)]

                    results_dic[ind + 1].append(
                        [results[index][ind + 1], region_new[0] - region_new[2] / 2.0, region_new[1] - region_new[3] / 2.0,
                         region_new[0] + region_new[2] / 2.0, region_new[1] + region_new[3] / 2.0])

                    regions.append([region_new[0] - region_new[2] / 2.0, region_new[1] - region_new[3] / 2.0, region_new[2],
                                    region_new[3], ind + 1, score])
            if len(regions) != 0:
                print(image_path+test_index+'.jpg')
                Data.show_rect(image_path+test_index+'.jpg', regions, test_index)
            else:
                print('There is no target')

            NMS_results = self.NMS(results_dic)
            if len(NMS_results) != 0:
                Data.show_rect(image_path+test_index+'.jpg', NMS_results, test_index )

            NMS_average_results = self.NMS_average(results_dic)
            if len(NMS_average_results) != 0:
                Data.show_rect(image_path+test_index+'.jpg', NMS_average_results, test_index )

    def NMS_IOU(self, vertice1, vertice2):  # verticle:[pro,xin,ymin,xmax,ymax]
        lu = np.maximum(vertice1[1:3], vertice2[1:3])
        rd = np.minimum(vertice1[3:], vertice2[3:])
        intersection = np.maximum(0.0, rd - lu)
        inter_square = intersection[0] * intersection[1]
        square1 = (vertice1[3] - vertice1[1]) * (vertice1[4] - vertice1[2])
        square2 = (vertice2[3] - vertice2[1]) * (vertice2[4] - vertice2[2])
        union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
        return np.clip(inter_square / union_square, 0.0, 1.0)

    def NMS(self, result_dic):
        final_result = []
        for cls_ind, cls_collect in result_dic.items():
            cls_collect = sorted(cls_collect, reverse=True)
            for i in range(len(cls_collect) - 1):
                for j in range(len(cls_collect) - 1, i, -1):
                    if self.NMS_IOU(cls_collect[i], cls_collect[j]) > cfg.NMS_threshold:
                        del cls_collect[j]

            for each_result in cls_collect:
                final_result.append(
                    [each_result[1], each_result[2], each_result[3] - each_result[1], each_result[4] - each_result[2],
                     cls_ind, each_result[0]])
        return final_result

    def NMS_average(self, result_dic):
        final_result = []
        for cls_ind, cls_collect in result_dic.items():
            cls_collect = sorted(cls_collect, reverse=True)
            for i in range(len(cls_collect) - 1):
                for j in range(len(cls_collect) - 1, i, -1):
                    if self.NMS_IOU(cls_collect[i], cls_collect[j]) > cfg.NMS_threshold:
                        cls_collect[i] = [(x + y) / 2.0 for (x, y) in zip(cls_collect[i], cls_collect[j])]
                        del cls_collect[j]

            for each_result in cls_collect:
                final_result.append(
                    [each_result[1], each_result[2], each_result[3] - each_result[1], each_result[4] - each_result[2],
                     cls_ind, each_result[0]])
        return final_result

if __name__ == '__main__':

    train = True
    if train:
        net = Net.Alexnet(is_training = True)
        data = Data.data()
        solver = Solver(net, data, is_training = True)
        solver.train()
    else:
        net = Net.Alexnet(is_training=False)
        data = Data.data()
        solver = Solver(net, data, is_training=False)
        solver.predict()













