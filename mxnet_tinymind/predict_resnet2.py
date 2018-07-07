# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import pandas as pd
from collections import namedtuple
import sys
sys.path.append('/data/mxnet/python')
import mxnet as mx

model_load = mx.model.FeedForward.load('./output/resnet18', 40)

test = mx.image.ImageIter(
             batch_size          = 1,
             data_shape          = (3, 64, 64),
             path_imglist        = './my.lst',
             path_root           = '../dataset/train',
             shuffle             = False,
             aug_list            = mx.image.CreateAugmenter(data_shape = (3, 64, 64),
                                                            resize = 64, 
                                                            mean = np.array([128.0, 128.0, 128.0]), 
                                                            std = np.array([256.0, 256.0, 256.0])
                                                            )
                          )

[prob, data1, label1] = model_load.predict(test, return_data=True)