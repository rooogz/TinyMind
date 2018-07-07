#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import random

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

def list_image(root, recursive, exts):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def make_list():
    image_list = list_image(root, recursive, exts)
    image_list = list(image_list)
    if shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + chunks - 1) // chunks
    for i in range(chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * train_ratio)
        sep_test = int(chunk_size * test_ratio)
        if train_ratio == 1.0:
            write_list(prefix + str_chunk + '.lst', chunk)
        else:
            if test_ratio:
                write_list(prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if train_ratio + test_ratio < 1.0:
                write_list(prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])

if __name__ == '__main__':
    chunks = 1 #默认
    recursive = True #默认
    root = './train_valid' #数据根目录
    shuffle = True #是否打乱数据
    train_ratio = 0.9 #训练集占总的比例，验证集占1-train_ratio
    lst_dir = './lst/'
    prefix = lst_dir + 'my' #生成lst文件放的路径及前缀
    test_ratio = 0 #不划分测试集
    exts = ['.jpeg', '.jpg', '.png'] #包含的图像格式
    if not os.path.isdir(lst_dir):
        os.mkdir(lst_dir)
    make_list()
    print('----done----')
   