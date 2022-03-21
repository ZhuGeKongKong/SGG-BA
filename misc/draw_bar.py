import pandas as pd
import nummpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import json
plt.style.use('fivethirtyeight')
import seaborn as sns

sns.set_style({'font.sans-serif':['simhei','Arial']}) 
results_bl = torch.load('/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/transformer_predcls_float32_epoch16_batch16/inference_test/VG_stanford_filtered_with_attribute_test/result_dict.pytorch')
results_g2s = torch.load('/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/transformer_predcls_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch')
mean_recall_bl = results_bl['predcls_mean_recall_list'][20]
mean_recall_g2s = results_g2s['predcls_mean_recall_list'][20]
vg_dict = json.load(open("/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json",'r'))
idx2pred = vg_dict['idx_to_predicate']
pred2idx = vg_dict['predicate_to_idx']
pred_count = vg_dict['predicate_count']
data=pd.DataFrame(np.random.rand(30,2)*1000,
                  columns=['Transformer','Transformer(G2S+ConfMat)'],
                  index=pd.period_range('2019-8-1','2019-8-30'))