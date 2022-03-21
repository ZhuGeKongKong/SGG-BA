import torch
import numpy as np
import json

vg_dict_info = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri-info.json','r'))
#vg_dict_info['predicate_to_idx']['__background__'] = 0
predicate_to_ind = vg_dict_info['predicate_to_idx']
ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

predicates_vg_info = vg_dict_info['predicate_information']
pred_info_arr = []
for i in range(len(ind_to_predicates)):
    pred_i = ind_to_predicates[i]
    if pred_i in predicates_vg_info:
        pred_info_arr.append(predicates_vg_info[pred_i])
    else:
        pred_info_arr.append(0.0)
pred_info_arr = np.array(pred_info_arr)
wiki_dict_info = json.load(open('./datasets/vg/WIKIPEDIA-info.json','r'))
predicates_wiki_info = wiki_dict_info['predicate_wiki_information']
pred_wiki_info_arr = []
for i in range(len(ind_to_predicates)):
    pred_i = ind_to_predicates[i]
    if pred_i in predicates_wiki_info:
        pred_wiki_info_arr.append(predicates_wiki_info[pred_i])
    else:
        pred_wiki_info_arr.append(0.0)
pred_wiki_info_arr = np.array(pred_wiki_info_arr)

task_name = "sgdet"
model_name = "transformer_"+task_name+"_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16"

eval_res = torch.load("./checkpoints/"+model_name+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch",map_location=torch.device("cpu"))
print("mean recall@20: ",eval_res[task_name+'_mean_recall'][20], "50: ",eval_res[task_name+'_mean_recall'][50], 
"100: ", eval_res[task_name+'_mean_recall'][100])
#print("content information vg 20: ",np.mean(eval_res['predcls_recallinfo'][20]), "50: ",np.mean(eval_res['predcls_recallinfo'][50]), "100: ", np.mean(eval_res['predcls_recallinfo'][100]))
#print("content information wiki 20: ",np.mean(eval_res['predcls_recallwikiinfo'][20]), "50: ",np.mean(eval_res['predcls_recallwikiinfo'][50]), "100: ", np.mean(eval_res['predcls_recallwikiinfo'][100]))

mean_recall_list = eval_res[task_name+'_mean_recall_list']
mricvg = {}
mricwk = {}

for i in [20,50,100]:
    print("mean recall@",i)
    print("hahahahaha")
    #print(np.array(mean_recall_list[i]))
    #print(np.array(pred_info_arr))
    print("mean_recall_list: ", np.array(mean_recall_list[i]).shape)
    print("pred_info_arr: ", np.array(pred_info_arr).shape)
    mricvg[i] = pred_info_arr * np.array(mean_recall_list[i])
    mricwk[i] = pred_wiki_info_arr * np.array(mean_recall_list[i])
print("content information mean recall vg 20: ",np.mean(mricvg[20]), "50: ",np.mean(mricvg[50]), "100: ", np.mean(mricvg[100]))
print("content information mean recall wiki 20: ",np.mean(mricwk[20]), "50: ",np.mean(mricwk[50]), "100: ", np.mean(mricwk[100]))
