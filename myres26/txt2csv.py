# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

f = open('submit.txt', 'r').readlines()
f = f[1:]
num = 0
save_arr = np.empty((len(f), 2), dtype=np.str)
save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
for line in f:
    img_name, predict = line.split('\t')
    save_arr.values[num, 0] = str(img_name)
    save_arr.values[num, 1] = predict[0:5]
    num += 1
save_arr.to_csv('submit.csv', decimal=',', encoding='utf-8', index=False, index_label=False)