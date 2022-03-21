import json
import numpy as np
import os
from tqdm import tqdm
OUTPUT_DIR = "/mnt/data1/guoyuyu/datasets/gqa/model/sgg_benchmark/checkpoints/"
def filter_triplets(inputname, outputname):

    with open(inputname, "r") as pred_file:
        pred_triplets = json.load(pred_file)
    pred_triplets_filter = {}
    for i in tqdm(pred_triplets):
        current_dict = pred_triplets[i]

        rel_pairs  = np.array(current_dict['rel_pairs'])
        rel_labels = np.array(current_dict['rel_labels'])
        rel_scores = np.array(current_dict['rel_scores'])
        bbox = current_dict['bbox']
        bbox_labels = current_dict['bbox_labels']
        bbox_scores = np.array(current_dict['bbox_scores'])
        triplets_scores = bbox_scores[rel_pairs[:,0]] * bbox_scores[rel_pairs[:,1]] * rel_scores
        triplets_sort_ind = np.argsort(0.0 - np.array(triplets_scores))

        max_num_triplets = min(128, len(rel_pairs))
        triplets_filter_ind = triplets_sort_ind[:max_num_triplets]
        triplets_scores_filter = np.array(triplets_scores)[triplets_filter_ind].tolist()
        rel_pairs_filter = np.array(rel_pairs)[triplets_filter_ind].tolist()
        rel_labels_filter = np.array(rel_labels)[triplets_filter_ind].tolist()
        rel_scores_filter = np.array(rel_scores)[triplets_filter_ind].tolist()
        current_dict_filter = {}
        current_dict_filter['bbox'] = current_dict['bbox']
        current_dict_filter['bbox_labels'] = current_dict['bbox_labels']
        current_dict_filter['bbox_scores'] = current_dict['bbox_scores']
        current_dict_filter['triplets_scores'] = triplets_scores_filter
        current_dict_filter['rel_pairs'] = rel_pairs_filter
        current_dict_filter['rel_labels'] = rel_labels_filter
        current_dict_filter['rel_scores'] = rel_scores_filter
        pred_triplets_filter[i] = current_dict_filter
    with open(outputname, 'w') as outfile:
        json.dump(pred_triplets_filter, outfile)

if __name__ == "__main__":
    exp_name = "/transformer_sgdet_wobias_woleftright_newdict_ba_3k/"
    input_name = OUTPUT_DIR+exp_name+'/extract_gqa_test_sgg/'+'custom_prediction.json'
    output_name = OUTPUT_DIR+exp_name+'/extract_gqa_test_sgg/'+'custom_prediction_top100.json'
    filter_triplets(input_name, output_name)
    input_name = OUTPUT_DIR+exp_name+'/extract_gqa_train_sgg/'+'custom_prediction.json'
    output_name = OUTPUT_DIR+exp_name+'/extract_gqa_train_sgg/'+'custom_prediction_top100.json'
    filter_triplets(input_name, output_name)

