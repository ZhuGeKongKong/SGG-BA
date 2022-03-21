import h5py
import json
import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.visual_genome import load_image_filenames, clip_to_image, load_info
from shutil import copyfile
import os
tgt_img_ind = 22182
split = 'train'
num_val_im = 5000
BOX_SCALE = 1024





triplets = [['plate', 'on', 'table'],['plate', 'on', 'table'],['plate', 'on', 'table']]

ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(
    "datasets/vg/VG-SGG-dicts-with-attri.json")
filenames, img_info = load_image_filenames("datasets/vg/VG_100K", "datasets/vg/image_data.json")

filter_empty_rels =True
filter_non_overlap = True
bias_model_res = torch.load(
    "/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/transformer_predcls_float32_epoch16_batch16/inference_train1/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch",
    map_location=torch.device("cpu"))

roidb_file = "/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-with-attri.h5"
roi_h5 = h5py.File(roidb_file, 'r')
vg_dict = json.load(open("/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json","r"))
ind2label = vg_dict['idx_to_label']
ind2pred = vg_dict['idx_to_predicate']

label_list = []
for i in range(len(ind2label)):
    label_list.append(ind2label[str(i+1)])
print("labels :",label_list)
pred_list = []
for i in range(len(ind2pred)):
        pred_list.append(ind2pred[str(i + 1)])
print("predicates :",pred_list)
data_split = roi_h5['split'][:]

split_flag = 2 if split == 'test' else 0
split_mask = data_split == split_flag

# Filter out images without bounding boxes
split_mask &= roi_h5['img_to_first_box'][:] >= 0
split_mask &= roi_h5['img_to_first_rel'][:] >= 0
image_index = np.where(split_mask)[0]
if split == 'val':
    image_index = image_index[:num_val_im]
elif split == 'train':
    image_index = image_index[num_val_im:]
split_mask = np.zeros_like(data_split).astype(bool)
split_mask[image_index] = True

all_labels = roi_h5['labels'][:, 0]
all_attributes = roi_h5['attributes'][:, :]
all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
assert np.all(all_boxes[:, :2] >= 0)  # sanity check
assert np.all(all_boxes[:, 2:] > 0)  # no empty box

# convert from xc, yc, w, h to x1, y1, x2, y2
all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

im_to_first_box = roi_h5['img_to_first_box'][split_mask]
im_to_last_box = roi_h5['img_to_last_box'][split_mask]
im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

# load relation labels
_relations = roi_h5['relationships'][:]
_relation_predicates = roi_h5['predicates'][:, 0]

assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

# Get everything by image.
boxes = []
gt_classes = []
gt_attributes = []
relationships = []

for i in range(len(image_index)):
    i_obj_start = im_to_first_box[i]
    i_obj_end = im_to_last_box[i]
    i_rel_start = im_to_first_rel[i]
    i_rel_end = im_to_last_rel[i]

    boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
    gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
    gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

    if i_rel_start >= 0:
        predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
        obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
        assert np.all(obj_idx >= 0)
        assert np.all(obj_idx < boxes_i.shape[0])
        rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
    else:
        assert not filter_empty_rels
        rels = np.zeros((0, 3), dtype=np.int32)

    if filter_non_overlap:
        assert split == 'train'
        # construct BoxList object to apply boxlist_iou method
        # give a useless (height=0, width=0)
        boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
        inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
        rel_overs = inters[rels[:, 0], rels[:, 1]]
        inc = np.where(rel_overs > 0.0)[0]

        if inc.size > 0:
            rels = rels[inc]
        else:
            split_mask[image_index[i]] = 0
            continue

    img_size = [img_info[image_index[i]]['width'], img_info[image_index[i]]['height']]
    keep = clip_to_image(boxes_i, img_size)
    if keep.sum() == 0:
        print('ignore img index: ', image_index[i])
        split_mask[image_index[i]] = 0
        continue
    pred_rel_scores = bias_model_res['predictions'][i].get_field('pred_rel_scores').detach().cpu().numpy()
    rel_pair_idxs = bias_model_res['predictions'][i].get_field('rel_pair_idxs').long().detach().cpu().numpy()
    pred_obj_scores = bias_model_res['predictions'][i].get_field('pred_scores').detach().cpu().numpy()
    find_tri = 0
    find_obj = 0
    for j in range(len(gt_classes_i)):
        obj = ind_to_classes[gt_classes_i[j]]
        if 'fork' == obj:
            find_obj = find_obj + 1
    for j in range(len(rels)):
        rel_j = rels[j]
        rel_j_pred = ind_to_predicates[rel_j[2]]
        rel_j_sub = ind_to_classes[gt_classes_i[rel_j[0]]]
        rel_j_obj = ind_to_classes[gt_classes_i[rel_j[1]]]
        k = 0
        if rel_j_sub == triplets[k][0] and rel_j_pred == triplets[k][1] and rel_j_obj == triplets[k][2]:
            find_tri = find_tri + 1
        if len(gt_classes_i) == len(pred_obj_scores):
            so_ind = rel_j[:2]
            pred_ind = np.where(np.sum(np.abs(so_ind[None, :] - rel_pair_idxs), -1) == 0)[0][0]
            rel_dist_j = pred_rel_scores[pred_ind][rel_j[2]]
    if find_tri and find_obj:
        print("file name: ", filenames[image_index[i]], "triplets: ", triplets[k])

        (filepath, tempfilename) = os.path.split(filenames[image_index[i]])
        target = os.path.join("./datasets/vg/select/", tempfilename)
        copyfile(filenames[image_index[i]], target)
    boxes.append(boxes_i)
    gt_classes.append(gt_classes_i)
    gt_attributes.append(gt_attributes_i)
    relationships.append(rels)





tgt_data_ind = image_index[tgt_img_ind]
print("image index in VG dataset: ", tgt_data_ind)
first_box_ind = roi_h5['img_to_first_box'][tgt_data_ind]
last_box_ind = roi_h5['img_to_last_box'][tgt_data_ind]
print("first box index: ", first_box_ind, "last box index: ", last_box_ind)
print("active_object_mask : ", roi_h5['active_object_mask'][first_box_ind:last_box_ind+1])
print("boxes_1024: ", roi_h5['boxes_1024'][first_box_ind:last_box_ind+1])
print("boxes_512: ", roi_h5['boxes_512'][first_box_ind:last_box_ind+1])
print("object label ind: ", roi_h5['labels'][first_box_ind:last_box_ind+1])
print("object label name: ",[ ind2label[str(i[0])] for i in roi_h5['labels'][first_box_ind:last_box_ind+1]])
first_rel_ind = roi_h5['img_to_first_rel'][tgt_data_ind]
last_rel_ind = roi_h5['img_to_last_rel'][tgt_data_ind]
print("first relationships index: ", first_rel_ind, "last relationships index: ", last_rel_ind)
print("predicate ind: ",roi_h5['predicates'][first_rel_ind:last_rel_ind+1])
print("predicate name: ",[ ind2pred[str(i[0])] for i in roi_h5['predicates'][first_rel_ind:last_rel_ind+1]])
print("relationships: ",roi_h5['relationships'][first_rel_ind:last_rel_ind+1])


