import torch
import numpy as np
import pandas as pd
import csv
import json

vg_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json','r'))
vg_id2pred = vg_dict['idx_to_predicate']
vg_pred2id = vg_dict['predicate_to_idx']
pred_count = vg_dict['predicate_count']
pred_count_sort = sorted(pred_count.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
baseline_name = 'transformer_predcls_float32_epoch16_batch16'
blru = 'transformer_predcls_dist15_2k_FixPModel_CleanH_Lr1e3_B16'
blra = 'transformer_predcls_TopDist15_TopBLMaxDist2k_FixPModel_lr1e3_B16'
blra_sacm = 'transformer_predcls_TopDist15_TopBLMaxDist2k_FixPModel_lr1e3_B16_FBLMCMat'
blra_sabg = 'transformer_predcls_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16'
model_baseline_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/"+baseline_name+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
model_baseline_recall = model_baseline_res['predcls_mean_recall_list'][20]
model_blru_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/"+blru+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
model_blru_recall = model_blru_res['predcls_mean_recall_list'][20]
model_blra_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/"+blra+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
model_blra_recall = model_blra_res['predcls_mean_recall_list'][20] 
model_blra_sacm_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/"+blra_sacm+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
model_blra_sacm_recall = model_blra_sacm_res['predcls_mean_recall_list'][20]
model_blra_sabg_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/"+blra_sabg+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
model_blra_sabg_recall = model_blra_sabg_res['predcls_mean_recall_list'][20]

def pred_sort_recall(model_recall, pred_count_sort):
    pred_sort_re_dict = []
    for pred_i in pred_count_sort:
        pred_sort_re_dict.append([pred_i[0], model_recall[int(vg_pred2id[pred_i[0]])-1]])
    return pred_sort_re_dict
print(len(model_baseline_recall))
model_baseline_recall_sort = pred_sort_recall(model_baseline_recall, pred_count_sort)
model_blru_recall_sort = pred_sort_recall(model_blru_recall, pred_count_sort)
model_blra_recall_sort = pred_sort_recall(model_blra_recall, pred_count_sort)
model_blra_sacm_recall_sort = pred_sort_recall(model_blra_sacm_recall, pred_count_sort)
model_blra_sabg_recall_sort = pred_sort_recall(model_blra_sabg_recall, pred_count_sort)

name=['Predicates','Transformer', 'Transformer+BLRU', 'Transformer+BLRA', 'Transformer+BLRA+SACM', 'Transformer+BLRA+SABG']
csv_data = []
for i in range(len(model_baseline_recall_sort)):
    #csv_data.append([model_pre_re_sort[i][0],model_pre_re_sort[i][1], model_g_re_sort[i][1], model_g2s_re_sort[i][1]])
    csv_data.append([model_baseline_recall_sort[i][0],
                     model_baseline_recall_sort[i][1],
                     model_blru_recall_sort[i][1], 
                     model_blra_recall_sort[i][1], 
                     model_blra_sacm_recall_sort[i][1],
                     model_blra_sabg_recall_sort[i][1]])
test=pd.DataFrame(columns=name,data=csv_data)
test.to_csv('./misc/predicate_recall_vsall.csv',encoding='gbk')
        
    