import numpy as np
import json 
vg_dict_info = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json','r'))
predicate_count = vg_dict_info['predicate_count']
pred_count_sum = 0
for i in predicate_count:
    pred_count_sum = pred_count_sum + predicate_count[i]
print("pred_count_sum: ", pred_count_sum)
pred_info = {}
for i in predicate_count:
    pred_info[i] = 0.0 - np.log(float(predicate_count[i]/pred_count_sum))#/np.log(float(10))
print("pred_info: ", pred_info)
    
    