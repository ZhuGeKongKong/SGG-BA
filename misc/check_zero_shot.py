import torch
import json
import numpy as np
import pandas as pd
vg_dict_info = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri-info.json','r'))
keep_pred = ['above', 'at', 'attached to', 'behind', 'for', \
'hanging from', 'has', 'holding', 'in', 'in front of', 'near', 'of', \
'on', 'over', 'sitting on', 'standing on', 'under', 'wearing', 'wears', \
'with']
pred2ind = vg_dict_info['predicate_to_idx']
ind2pred = vg_dict_info['idx_to_predicate']
#checkpoints/transformer_predcls_top20cls_unseen_Mattrans/test
#checkpoints_best/transformer_predcls_float32_epoch16_batch16
res_model = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints/transformer_predcls_top20cls_unseen_PredW2VClsGCNEmbed_PreTra_MSE/test/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
zero_pred_recall = {}
for rk in [20,50,100]:
    mean_reall = res_model["predcls_mean_recall_list"][rk]
    print("not zero: ",np.sum(np.array(mean_reall)!=0))
    keep_pred_ind = []
    zero_pred_recall_rk = []
    zero_pred = []
    for i in keep_pred:
        keep_pred_ind.append(pred2ind[i])
    for i in range(len(mean_reall)):
        if i+1 not in keep_pred_ind:
            print("pred: ", ind2pred[str(i+1)],"recall "+str(rk)+": ", mean_reall[i])
            zero_pred.append(ind2pred[str(i+1)])
            zero_pred_recall_rk.append(mean_reall[i])
    zero_pred.append("mean")
    zero_pred_recall_rk.append(np.array(zero_pred_recall_rk).mean())
    zero_pred_recall[rk] = zero_pred_recall_rk
    print(np.array(zero_pred_recall[rk]).mean())
dataframe = pd.DataFrame({'predicates':zero_pred,'R@20':zero_pred_recall[20],'R@50':zero_pred_recall[50],'R@100':zero_pred_recall[100]})
dataframe.to_csv("tail_pred_recall_transformer_gcn.csv",index=False,sep=',')

    



