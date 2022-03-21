import wikipedia
import json
import numpy as np
VG_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
pred_list = VG_dict['predicate_to_idx']
pred_count_wikipedia_all = {}
for i in pred_list:
    pred_count_wikipedia_all[i] = len(wikipedia.search(i,results=np.inf))