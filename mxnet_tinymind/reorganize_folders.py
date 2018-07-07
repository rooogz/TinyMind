# -*- coding: utf-8 -*-
import os
import shutil

#%% 使用mxnet-gluon之前需要整理文件夹
def reorganize_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # 读取训练数据标签。
    f = open(os.path.join(data_dir, label_file), 'r')
    # 跳过文件头行（栏名称）。
    lines = f.readlines() #[1:]
    num_train = len(lines)
    tokens = [l.rstrip().split(' ') for l in lines]
    idx_label = dict(((idx, label) for idx, label in tokens)) #文件名字及类
    labels = set(idx_label.values())
#    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int(num_train * (1 - valid_ratio)) #总train的个数
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels) #每一类train的个数
    label_count = dict()
    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))
    # 整理训练和验证集。
    for train_file in lines:
        idx = train_file.split(' ')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, idx),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, idx),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, idx),
                        os.path.join(data_dir, input_dir, 'valid', label))
    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
#%%
#--------未整理时各个文件夹目录如下：----------
#dataset/train/已知标签的图片
#       /test1/未知标签的图片
#       /trainLabels.csv

#--------整理后各个文件夹目录如下：------- ----
#dataset/train/已知标签的图片
#       /test1/未知标签的图片
#       /trainLabels.csv
#       /train_valid_test/test/unknown/未知标签的图片
#                        /train/dog/所有dog类的图片
#                              /cat/所有cat类的图片
#                              /......
#                        /train_valid/dog/所有dog类的图片
#                                    /cat/所有cat类的图片
#                                    /......
#                        /valid/dog/所有dog类的图片
#                              /cat/所有cat类的图片
#                              /......
if __name__ == '__main__':
    data_dir = './' #数据的根目录
    label_file = 'all_data2.txt' #标签文件
    train_dir = 'train' #整理前/后训练集文件夹名字。所有数据包括训练和验证集
    test_dir = 'test1' #整理前/后测试集文件夹名字，不知道标签，待预测的数据
    input_dir = 'train_valid_test' #整理后数据的文件夹根目录
    valid_ratio = 4000/40000.0 #划分验证集占总训练集的比例
    reorganize_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)