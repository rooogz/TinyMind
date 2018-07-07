# -*- coding: utf-8 -*-

import logging
logging.getLogger().setLevel(logging.INFO)
import sys, os
import numpy as np
import argparse
sys.path.append('/data/mxnet/python')
try:
    import mxnet as mx
except ImportError:
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "../../python"))
    import mxnet as mx
from mxnet import nd
import resnet

#%% 设置参数
parser = argparse.ArgumentParser(description='set parameters',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size'  , type=int  , default=128           , help='batch_size')
parser.add_argument('--num_epoch'   , type=int  , default=40            , help='epoch number')
parser.add_argument('--lr'          , type=float, default=0.05          , help='learning rate')
parser.add_argument('--step_value'  , type=int  , default=10            , help='lr decay, every step_value epoch')
parser.add_argument('--display'     , type=int  , default=100           , help='print the accuracy every display')
parser.add_argument('--model_prefix', type=str  , default='resnet18'    , help='prefix of saved model')
parser.add_argument('--num_classes' , type=int  , default=100           , help='class number')
parser.add_argument('--wd'          , type=float, default=1e-5          , help='weight decay')
parser.add_argument('--momentum'    , type=float, default=0.9           , help='momentum')
parser.add_argument('--data_shape'  , type=str  , default='3,64,64'     , help='data argument: data shape')
parser.add_argument('--resize'      , type=int  , default=64            , help='data argument: resize image')
parser.add_argument('--rand_crop'   , type=bool , default=False         , help='data argument: rand crop')
parser.add_argument('--rand_resize' , type=bool , default=False         , help='data argument: rand resize') 
parser.add_argument('--rand_mirror' , type=bool , default=False         , help='data argument: rand mirror')
parser.add_argument('--mean'        , type=str  , default='0,0,0'       , help='data mean value')
parser.add_argument('--std'         , type=str  , default='1,1,1'       , help='data std value')
parser.add_argument('--brightness'  , type=float, default=0             , help='data argument')
parser.add_argument('--contrast'    , type=float, default=0             , help='data argument') 
parser.add_argument('--saturation'  , type=float, default=0             , help='data argument') 
parser.add_argument('--hue'         , type=float, default=0             , help='data argument') 
parser.add_argument('--pca_noise'   , type=float, default=0             , help='data argument')
parser.add_argument('--rand_gray'   , type=float, default=0             , help='data argument')
parser.add_argument('--inter_method', type=int  , default=2             , help='data argument')

parser.add_argument('--work_dir'    , type=str  , default='/aaaa/'      , help='no using')
parser.add_argument('--data_dir'    , type=str  , default='/aaaa/'      , help='no using')
parser.add_argument('--output_dir'  , type=str  , default='/aaaa/'      , help='no using')
parser.add_argument('--num_gpus'    , type=int  , default=1             , help='no using')

args = parser.parse_args()
args.data_shape = tuple([int(l) for l in args.data_shape.split(',')]) # str to tuple
args.mean = np.array([float(l) for l in args.mean.split(',')]) #str to np.array
args.std = np.array([float(l) for l in args.std.split(',')]) #str to np.array

path_train_imglist = './lst/my_train.lst'
path_val_imglist   = './lst/my_val.lst'
path_root          = './train_valid'
snapshot           = './output/'

#batch_size         = 128
#num_epoch          = 40
#lr                 = 0.05
#step_value         = 10 #每隔step_value*epoch 个迭代，学习率衰减一次
#display            = 100
#model_prefix       = 'renet18' #保存的模型
#num_classes        = 100
#wd                 = 1e-5
#momentum           = 0.9
#data_shape         = (3,64,64)
#resize             = 64
#rand_crop          = False
#rand_resize        = False
#rand_mirror        = False
#mean               = np.array([0,0,0])
#std                = np.array([1,1,1])
#brightness         = 0
#contrast           = 0
#saturation         = 0 
#hue                = 0
#pca_noise          = 0 
#rand_gray          = 0
#inter_method       = 2

with open(path_train_imglist, 'r') as f:
    num_sample     = len(f.readlines())
one_epoch          = num_sample // args.batch_size
lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=int(args.step_value*one_epoch), factor=0.1) #lr*pow(factor,floor(num_update/step))经过1500次参数更新后，学习率变为 lr×0.1。经过3000次参数更新之后，学习率变为 lr×0.1×0.1
                                                                                                 #每隔int(step_value*one_epoch)次迭代(step_value个epoch)，lr*0.1

optimizer_params   = {
        'learning_rate'    : args.lr,
        'wd'               : args.wd,
        'momentum'         : args.momentum,
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
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=args.num_classes)
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
devs = try_gpu()
print('the device is: ' + str(devs))
if not os.path.isdir(snapshot):
    os.mkdir(snapshot)
train_iter = mx.image.ImageIter(
             batch_size          = args.batch_size,
             data_shape          = args.data_shape,
             label_width         = 1,
             path_imglist        = path_train_imglist,
             path_root           = path_root,
             part_index          = 0,
             shuffle             = True,
             data_name           = 'data',
             label_name          = 'softmax_label',
             aug_list            = mx.image.CreateAugmenter(data_shape  =args.data_shape , resize      =args.resize     , rand_crop   =args.rand_crop  , 
                                                            rand_resize =args.rand_resize, rand_mirror =args.rand_mirror, mean        =args.mean       , 
                                                            std         =args.std        , brightness  =args.brightness , contrast    =args.contrast   ,
                                                            saturation  =args.saturation , hue         =args.hue        , pca_noise   =args.pca_noise  , 
                                                            rand_gray   =args.rand_gray  , inter_method=args.inter_method))

val_iter = mx.image.ImageIter(
             batch_size          = args.batch_size,
             data_shape          = args.data_shape,
             label_width         = 1,
             path_imglist        = path_val_imglist,
             path_root           = path_root,
             part_index          = 0,
             shuffle             = True,
             data_name           = 'data',
             label_name          = 'softmax_label',
             aug_list            = mx.image.CreateAugmenter(data_shape =args.data_shape , resize      =args.resize, mean =args.mean, 
                                                            std        =args.std        , inter_method=args.inter_method))


# 定义模型
# create a trainable module on GPU 0
#symbol = lenet()
symbol = resnet.get_symbol(num_classes=args.num_classes, num_layers=18, image_shape=str(args.data_shape[0])+','+str(args.data_shape[1])+','+str(args.data_shape[2]))

model = mx.mod.Module(symbol=symbol, context=devs)
# 保存模型
checkpoint = mx.callback.do_checkpoint(snapshot + args.model_prefix)
# 训练模型
model.fit(train_iter,
          eval_data=val_iter,
          optimizer='sgd',
          optimizer_params=optimizer_params,
          eval_metric='acc',
          batch_end_callback = mx.callback.Speedometer(args.batch_size, args.display),
          num_epoch=args.num_epoch,
          epoch_end_callback=checkpoint)





