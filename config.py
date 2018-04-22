import os
All_list = r'./Data/all_list.txt'
Train_list = r'./Data/train_list.txt'
Valid_list = r'./Data/valid_list.txt'
Annotation_path = r'./Data/Annotations'
Images_path = r'./Data/Images/'
Test_output = r'./Test_output'
Processed_path = r'./Data/Processed'
Weights_file=r'./train_alexnet/Alexnet'

Classes=['bus']
Class_num = 2#因为有背景在 所以比Classes多一项
Image_w = 1600
Image_h = 1200
Roi_num = 600#防止候选框太多所做的限定,在我的项目里我取的是合格框数量（150）的4倍

Batch_size = 1
Max_iter =  100000
Summary_iter = 10
Save_iter = 500

Initial_learning_rate = 0.00001
Decay_rate = 0.8
Decay_iter = 900
Staircase = True
Roi_threshold = 0.7
NMS_threshold = 0.05
