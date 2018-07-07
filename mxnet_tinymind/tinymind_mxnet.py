# -*- coding: utf-8 -*-

import logging
logging.getLogger().setLevel(logging.INFO)
import sys, os
sys.path.append('/data/mxnet/python')
import mxnet as mx
from mxnet import nd
import resnet

#%% 设置参数
path_train_imglist = './lst/my_train.lst'
path_val_imglist   = './lst/my_val.lst'
path_root          = '../dataset/train' #'./train_valid'
snapshot           = './output/'
model_prefix       = snapshot + 'resnet18' #保存的模型
batch_size         = 10
num_epoch          = 50
lr                 = 0.01
step_value         = 5 #每隔step_value*epoch 个迭代，学习率衰减一次
display            = 100
num_classes        = 100
wd                 = 1e-5
momentum           = 0.9
data_shape         = (3,64,64)
resize             = 64
with open(path_train_imglist, 'r') as f:
    num_sample     = len(f.readlines())
one_epoch          = num_sample // batch_size
lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=int(step_value*one_epoch), factor=0.1) #lr*pow(factor,floor(num_update/step))经过1500次参数更新后，学习率变为 lr×0.1。经过3000次参数更新之后，学习率变为 lr×0.1×0.1
                                                                                                 #每隔int(step_value*one_epoch)次迭代(step_value个epoch)，lr*0.1

optimizer_params   = {
        'learning_rate'    : lr,
        'wd'               : wd,
        'momentum'         : momentum,
        'multi_precision'  : True,
        'lr_scheduler'     : lr_scheduler,
#        'begin_epoch'      : 0,
#        'updates_per_epoch': epoch_size,
#        'batch_scale'      : batch_scale,
#        'num_epochs'       :num_epochs
        }

#%% lenet
def lenet():
    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=num_classes)
    # softmax loss
    le = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return le

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
#%%# 导入数据并初始化迭代器
if not os.path.isdir(snapshot):
    os.mkdir(snapshot)
train_iter = mx.image.ImageIter(
             batch_size          = batch_size,
             data_shape          = data_shape,
             label_width         = 1,
             path_imglist        = path_train_imglist,
             path_root           = path_root,
             part_index          = 0,
             shuffle             = True,
             data_name           = 'data',
             label_name          = 'softmax_label',
             aug_list            = mx.image.CreateAugmenter(data_shape,resize=resize,rand_crop=False,rand_mirror=False,mean=True)) #mx.image.CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, pca_noise=0, inter_method=2)

val_iter = mx.image.ImageIter(
             batch_size          = batch_size,
             data_shape          = data_shape,
             label_width         = 1,
             path_imglist        = path_val_imglist,
             path_root           = path_root,
             part_index          = 0,
             shuffle             = True,
             data_name           = 'data',
             label_name          = 'softmax_label',
             aug_list            = mx.image.CreateAugmenter(data_shape,resize=resize,rand_crop=False,rand_mirror=False,mean=True)) #mx.image.CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0, saturation=0, pca_noise=0, inter_method=2)


# 定义模型
# create a trainable module on GPU 0
#symbol = lenet()
symbol = resnet.get_symbol(num_classes=num_classes, num_layers=18, image_shape=str(data_shape[0])+','+str(data_shape[1])+','+str(data_shape[2]))
model = mx.mod.Module(symbol=symbol, context=try_gpu())
# 保存模型
checkpoint = mx.callback.do_checkpoint(model_prefix)
# 训练模型
model.fit(train_iter,
          eval_data=val_iter,
          optimizer='sgd',
          optimizer_params=optimizer_params,
          eval_metric='acc',
          batch_end_callback = mx.callback.Speedometer(batch_size, display),
          num_epoch=num_epoch,
          epoch_end_callback=checkpoint)





