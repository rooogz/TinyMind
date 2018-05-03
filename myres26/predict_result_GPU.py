#coding=utf-8
import sys,os
import cv2
import numpy as np
import pandas as pd

img_size = 64

#caffe_root = '/data/caffe/'
#sys.path.insert(0, caffe_root + 'python')
import caffe
#os.chdir(caffe_root) #切换到该目录
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

test_dir = '/data/data/test1_process/' #test image path
net_file = '/data/deploy_mylenet.prototxt'
caffe_model = '/data/snapshot/mylenet2_iter_2000.caffemodel'
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) ##改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
#transformer.set_raw_scale('data', 255) # from [0,1] to [0,255],caffe.io.load_image()读进来的是RGB格式和0~1(float),cv2.imread()读入的是0-255。当用cv2.read()时不加这行
#transformer.set_channel_swap('data', (2,1,0)) #将RGB变换到BGR

test_list = os.listdir(test_dir)
print('----------- %s number is: %d -------------' % ('test image', len(test_list)))
num = 0
label_dict = {}
with open('/data/mydata/labe_dict.txt', 'r') as f:
    for line in f.readlines():
        chinese, label = line.split(' ')
        label_dict[int(label)] = chinese
                   
f = open('/data/output/submit.txt', 'w')
f.write('filename' + '\t' + 'label' + '\n')

save_arr = np.empty((len(test_list), 2), dtype=np.str)
save_arr = pd.DataFrame(save_arr, columns=['filename', 'lable'])
for img_name in test_list:
    img_dir = os.path.join(test_dir, img_name)
    
#    im = caffe.io.load_image(img_dir)
    im = cv2.imread(img_dir)
    im = cv2.resize(im, (img_size, img_size))
#    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #转为灰度
    im = im / 256.0
    width, height = im.shape[1],im.shape[0]
    
    # im
    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    out = net.forward()
    prob_all = net.blobs['prob'].data[0].flatten() #为0至所有类的概率

    top_k = prob_all.argsort()[-1:-6:-1] #top5,返回最大概率对应的位置(即标签)
    top_k = top_k.tolist()
    
    predict = label_dict[top_k[0]] + label_dict[top_k[1]] + label_dict[top_k[2]] + label_dict[top_k[3]] + label_dict[top_k[4]]
    f.write(img_name + '\t' + predict + '\n')
    save_arr.values[num, 0] = img_name
    save_arr.values[num, 1] = predict

    num += 1
    print('%d %s is %s' % (num, img_name, str(top_k[0])+';'+str(top_k[1])+';'+str(top_k[2])+';'+str(top_k[3])+';'+str(top_k[4])))
f.close()
#save_arr.to_csv('/data/output/submit.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
print('---------------predicted--------------')
