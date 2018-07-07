# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import pandas as pd
from collections import namedtuple
import sys
sys.path.append('/data/mxnet/python')
import mxnet as mx

#%%
def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def get_image(imgpath, show=False):
    # download and show the image
#    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
#    if img is None:
#         return None
#    if show:
#         plt.imshow(img)
#         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.imread(imgpath)
    width, height, channel = img.shape
    if width > height:
        rate = width / float(height)
        new_size = (int(64*rate), 64)
    else:
        rate =  height / float(width)
        new_size = (64, int(64*rate))
#    new_size = (64, 64)
    img = cv2.resize(img, new_size)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img[0,:,:] = (img[0,:,:] - mean_value[0])/std[0]
    img[1,:,:] = (img[1,:,:] - mean_value[1])/std[1]
    img[2,:,:] = (img[2,:,:] - mean_value[2])/std[2]
    img = img[np.newaxis, :]
    return img

#%%
data_path = '../dataset/'
mean_value = [128.0, 128.0, 128.0]
std = [256.0, 256.0, 256.0]
imglist = os.listdir(data_path + 'test1/')
sym, arg_params, aux_params = mx.model.load_checkpoint('./output/resnet18', 40)
mod = mx.mod.Module(symbol=sym, context=try_gpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,64,64))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

label_dict = {}
with open(data_path + 'labe_dict.txt', 'r') as f:
    for line in f.readlines():
        chinese, label = line.split(' ')
        label_dict[int(label)] = chinese

Batch = namedtuple('Batch', ['data'])

save_arr = np.empty((len(imglist), 2), dtype=np.str)
save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
num = 0
for img_name in imglist:
    img = get_image(data_path + 'test1/' +img_name)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob) 
    a = np.argsort(prob)[::-1] #概率由高到低排列时输出对应的位置(位置即标签)
    top_k = a[0:5] #top5
    predict = label_dict[top_k[0]] + label_dict[top_k[1]] + label_dict[top_k[2]] + label_dict[top_k[3]] + label_dict[top_k[4]]
    save_arr.values[num, 0] = img_name
    save_arr.values[num, 1] = predict

    num = num + 1 
    print('%d %s is %s %f' % (num, img_name, predict, prob[top_k[0]]))

save_arr.to_csv('./submit.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
print('--------done--------')

#%% 提取特征
## list the last 10 layers
#all_layers = sym.get_internals()
#all_layers.list_outputs()[-10:] #最后10个层
## 获取全连接前的那一层输出
#fe_sym = all_layers['flatten0_output']
#fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
#fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
#fe_mod.set_params(arg_params, aux_params)
#
#img = get_image('http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
#fe_mod.forward(Batch([mx.nd.array(img)]))
#features = fe_mod.get_outputs()[0].asnumpy()
#print(features)
#assert features.shape == (1, 2048)