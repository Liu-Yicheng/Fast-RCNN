import os
import xml.etree.ElementTree as ET
import cv2
import config as cfg
import codecs
import selectivesearch
import numpy as np
import math
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):

    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

def IOU(ver1, vertice2):  # ver1:[xmin, ymin, w,h], verticel2:[xmin, ymin, xmax, ymax]
    vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    lu = np.maximum(vertice1[0:2], vertice2[0:2])
    rd = np.minimum(vertice1[2:], vertice2[2:])
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]
    square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    return np.clip(inter_square / union_square, 0.0, 1.0)

def view_bar(message, num, total):

    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def show_rect(img_path, regions, test_index):
    ind_to_class = dict(zip(range(1, len(cfg.Classes) + 1), cfg.Classes))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y, w, h, cls_ind, score in regions:
        xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
        if xmin <= 0: xmin = 1
        if xmax >= cfg.Image_w: xmax = cfg.Image_w - 1
        if ymin <= 0: ymin = 1
        if ymax >= cfg.Image_h: ymax = cfg.Image_h - 1
        message = ind_to_class[cls_ind]
        rect = cv2.rectangle(
            img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message + ': ' + str(round(score, 2)), (xmin + 5, ymin - 20), font, 1, (255,255,255), 2)
    if not os.path.exists(cfg.Test_output):
        os.makedirs(cfg.Test_output)
    out_path = os.path.join(cfg.Test_output, test_index+'test.jpg')
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    plt.imshow(img)
    plt.show()

class data(object):

    def __init__(self, is_save=True):
        self.all_list = cfg.All_list
        self.train_list = cfg.Train_list
        self.valid_list = cfg.Valid_list
        self.annotation_path = cfg.Annotation_path
        self.images_path = cfg.Images_path
        self.processed_path = cfg.Processed_path

        self.classes = cfg.Classes
        self.class_num = cfg.Class_num
        self.class_to_ind = dict(zip(self.classes, range(1, len(self.classes) + 1)))
        self.image_w = cfg.Image_w
        self.image_h = cfg.Image_h

        self.is_save = is_save
        self.batch_size = cfg.Batch_size
        self.roi_threshold = cfg.Roi_threshold
        self.train_images_index = []
        self.cursor = 0
        self.epoch = 0

        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

        if len(os.listdir(self.processed_path)) == 0:
            self.generate_labels()

    def generate_labels(self):
        with codecs.open(self.all_list, 'r', 'utf-8') as f:
            lines = f.readlines()
            for num, image_idx in enumerate(lines):
                ground_truth_dic = self.load_annotation(image_idx)
                image_path = os.path.join(self.images_path, image_idx.strip() + '.jpg')
                img = cv2.imread(image_path)
                img_lbl, regions = selectivesearch.selective_search(img, scale=1000, sigma=0.9, min_size=1000)
                labels = []
                for r in regions:
                    x, y, w, h = r['rect']
                    proposal_vertice = [x + 1, y, x +  w, y +  h, w, h]
                    proposal_bbox = [x, y, (x + w - 1), (y + h - 1)]
                    label = np.zeros(self.class_num * 5 - 4, dtype=np.float32)  # 假设包括背景有5类，0：5是判断类别，5：5+4*4=21 是位置框信息
                    iou_val=0
                    for ground_truth, class_idx in ground_truth_dic.items():
                        #ground_truth = list(ground_truth)
                        xmin=(2*ground_truth[0]-ground_truth[2])/2.0
                        ymin=(2*ground_truth[1]-ground_truth[3])/2.0
                        ground_truth=[xmin,ymin,ground_truth[2],ground_truth[3]]
                        iou_val = IOU(ground_truth, proposal_bbox)
                        px = float(proposal_vertice[0]) + float(proposal_vertice[4] / 2.0)  # 中心点X
                        py = float(proposal_vertice[1]) + float(proposal_vertice[5] / 2.0)  # 中心点Y
                        pw = float(proposal_vertice[4])  # w
                        ph = float(proposal_vertice[5])  # h

                        gx = float(ground_truth[0])  # 中心点X
                        gy = float(ground_truth[1])  # 中心点Y
                        gw = float(ground_truth[2])  # W
                        gh = float(ground_truth[3])  # H

                        if iou_val < self.roi_threshold  :
                            label[0] = 1
                        elif iou_val > self.roi_threshold:
                            label[0] = 0
                            label[class_idx] = 1
                            label[self.class_num + (class_idx-1)*4 : self.class_num + (class_idx-1)*4 + 4] = \
                                [((gx - px) / pw), ((gy - py) / ph), (np.log(gw / pw)), (np.log(gh / ph))]
                            break
                    for i in range(len(proposal_bbox)):
                        proposal_bbox[i] = (proposal_bbox[i] / 16.0)
                    proposal_bbox.insert(0, 0)
                    proposal_bbox.insert(0, iou_val)
                    proposal_bbox.extend(label)
                    labels.append(proposal_bbox)
                view_bar("Process image of %s" % image_path, num + 1, len(lines))
                if self.is_save:
                    if not os.path.exists(self.processed_path):  os.makedirs(self.processed_path)
                    np.save((os.path.join(self.processed_path, image_idx.split('.')[0].strip())
                             + '_data.npy'), labels)

    def load_annotation(self, image_idx):
        iamge_annotion_path = os.path.join(self.annotation_path, image_idx.strip() + '.xml')
        tree = ET.parse(iamge_annotion_path)
        objs = tree.findall('object')
        labels = {}
        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            class_name = obj.find('name').text.lower().strip()
            if class_name != 'left':
                cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
                labels[tuple(boxes)] = cls_ind
        return labels

    def get_batch(self):

        if len(self.train_images_index) == 0:
            print('load train list----------')
            with open(self.train_list, 'r') as f:
                for line in f.readlines():
                    image_index = line.strip()
                    self.train_images_index.append(image_index)
                np.random.shuffle(self.train_images_index)

        images = []
        rois = []
        label = []
        for i in range(self.batch_size):
            images_path = os.path.join(self.images_path, self.train_images_index[self.cursor] + '.jpg')
            image = cv2.imread(images_path)
            images.append(image)
            labels = np.load(os.path.join(self.processed_path, self.train_images_index[self.cursor] + '_data.npy'))
            labels = sorted(labels.tolist(), reverse=True)
            select_num = min(cfg.Roi_num, len(labels))
            for rois_label in labels[0:select_num]:
                rois.append(
                        [rois_label[1] + i, int(rois_label[2])-1, int(rois_label[3])-1, int(rois_label[4])+1, int(rois_label[5])+1])
                label.append((rois_label[6:]))
            self.cursor += 1
            if self.cursor >= len(self.train_images_index):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.train_images_index)
        rois = np.array(rois)
        label = np.array(label)
        images = np.array(images)
        return images, rois, label

    def get_valid_batch(self, test_index):
        valid_images_index = test_index
        images_path = os.path.join(self.images_path, valid_images_index + '.jpg')

        images = cv2.imread(images_path)
        labels = np.load(os.path.join(self.processed_path, valid_images_index + '_data.npy'))
        labels = sorted(labels.tolist(), reverse=True)
        rois = []
        label = []
        for rois_label in labels:
            rois.append([rois_label[1], (rois_label[2]), (rois_label[3]), (rois_label[4]), (rois_label[5])])
            label.append((rois_label[6:]))
        rois = np.array(rois)
        label = np.array(label)
        return [images], rois, label
