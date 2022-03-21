import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import colorsys
import random
import os
from graphviz import Digraph
import matplotlib.pyplot as plt
from maskrcnn_benchmark.layers import nms as _box_nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList
model_name = 'transformer_predcls_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16'
pred_model_name = 'transformer_predcls_float32_epoch16_batch16'
pred_list = ["standing on", "sitting on", "looking at", "riding","holding", "eating"]
project_dir = ''
image_file = json.load(open('./datasets/vg/image_data.json'))
vocab_file = json.load(open('./datasets/vg/VG-SGG-dicts.json'))
pred2idx = vocab_file['predicate_to_idx']
pred_idx_list = []
for i in pred_list:
    pred_idx_list.append(pred2idx[i])
data_file = h5py.File('./datasets/vg/VG-SGG.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
idx2label = vocab_file['idx_to_label']
label2idx = vocab_file['label_to_idx']
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# load detected results
detected_origin_path = './checkpoints_best/'+str(model_name)+'/inference/VG_stanford_filtered_with_attribute_test/'
pre_detected_origin_path = './checkpoints_best/'+str(pred_model_name)+'/inference/VG_stanford_filtered_with_attribute_test/'
main_path = detected_origin_path
detected_origin_output_path = detected_origin_path + '/visualization_good/'
pre_detected_origin_output_path = pre_detected_origin_path + '/visualization_good/'
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder  ---: ", path)
    else:
        print("---  There is this folder!  ---")
# mkdir(output_path)

detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
pre_detected_origin_result = torch.load(pre_detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

# get image info by index

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def box_nms_gt(groundtruth):
    boxes = groundtruth.bbox
    labels = groundtruth.get_field('labels')
    print(labels)
    keep_all = []
    for i in range(1, 151):
        inds_i = (labels == i).nonzero().view(-1)
        #print(inds_i)
        boxes_i = boxes[inds_i, :].view(-1, 4)
        boxlist_for_class = BoxList(boxes_i, groundtruth.size, mode="xyxy")
        scores_i = torch.ones(boxes_i.size(0))
        boxlist_for_class.add_field("scores", scores_i)
        _, keep = boxlist_nms(
            boxlist_for_class, 0.5,
            score_field="scores"
        )
        if len(keep_all) == 0:
            keep_all = inds_i[keep]
        else:
            keep_all = torch.cat([keep_all, inds_i[keep]], 0)
    keep_all_r = []
    count_idx = 0
    for i in range(len(keep_all)):
        if idx2label[str(labels[keep_all[i]].numpy())] == 'flower':
            count_idx =count_idx + 1
            if count_idx > 3:
                continue
        keep_all_r.append(keep_all[i])
    keep_all = torch.LongTensor(np.array(keep_all_r))
    return keep_all

def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']


    # boxes
    #boxes = prediction.bbox
    boxes = groundtruth.bbox
    labels = groundtruth.get_field('labels')
    labels_list = []
    for i in labels:
        labels_list.append(idx2label[str(i.numpy())])
    keep_gt_indices = box_nms_gt(groundtruth)
    keep_gt_indices = keep_gt_indices.numpy()
    gt_old2new = {}
    min_are = 999999999
    min_ind = 0
    # for i, keep_i in enumerate(keep_gt_indices):
    #     if idx2label[str(int(labels[keep_i]))] == 'umbrella': #or idx2label[str(int(labels[keep_i]))] == 'rock':
    #         x1, y1, x2, y2 = boxes[keep_i]
    #         area = (x2-x1)*(y2-y1)
    #         if area<min_are:
    #             min_are = area
    #             min_ind = i
    # keep_gt_indices = np.delete(keep_gt_indices, min_ind)

    for i, keep_i in enumerate(keep_gt_indices):
        gt_old2new[keep_i] = i
    boxes = boxes[keep_gt_indices]
    labels = labels[keep_gt_indices]

    # object labels


    #labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]

    # pred_label_num = {}
    # pred_labels_list = prediction.get_field('pred_labels').tolist() #
    # pred_labels_new = []
    # for i in pred_labels_list:
    #     pred_tmp = idx2label[str(int(i))]
    #     if pred_tmp not in pred_label_num:
    #         pred_labels_new.append(pred_tmp)
    #         pred_label_num[pred_tmp] = 1
    #     else:
    #         pred_labels_new.append(str(pred_label_num[pred_tmp])+'-'+pred_tmp)
    #         pred_label_num[pred_tmp] = pred_label_num[pred_tmp] + 1
    pred_labels_list = labels.tolist()
    pred_labels = pred_labels_list
    #print(pred_labels)
    #pred_labels = ['{}-{}'.format(idx,idx2label[str(int(i))]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    pred_scores = prediction.get_field('pred_scores')#.tolist()
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    #gt_rels = groundtruth.get_field('relation_tuple').tolist()
    #gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    gt_rels = None
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs')
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)

    # mask = pred_rel_score > thres
    # pred_rel_score = pred_rel_score[mask]
    # pred_rel_label = pred_rel_label[mask]
 
    pred_rel_pair_tmp = []
    pred_rel_label_tmp = []
    pred_rel_score_tmp = []
    old2new_idx = {}
    for i,j,k in zip(pred_rel_pair, pred_rel_label, pred_rel_score):
        if int(i[0]) in keep_gt_indices and int(i[1]) in keep_gt_indices:
            pred_rel_score_tmp.append(k)
            pred_rel_label_tmp.append(j)
            pred_rel_pair_tmp.append(i.numpy())
    pred_rel_label = np.array(pred_rel_label_tmp)
    pred_rel_score = np.array(pred_rel_score_tmp)
    pred_rel_pair = np.array(pred_rel_pair_tmp)
    
    keep_box_idx = []
    #
    #print(np.sort(-pred_rel_score))
    #pred_rel_sort_ind = pred_rel_score > 0.2
    #pred_rel_sort_ind = np.argsort(-pred_rel_score)[:int(len(pred_rel_score)/3)]
    # if len(pred_rel_score) > 24:
    #     pred_rel_sort_ind = np.argsort(-pred_rel_score)[:24]
    # else:
    #     pred_rel_sort_ind = np.argsort(-pred_rel_score)
    # pred_rel_score = pred_rel_score[pred_rel_sort_ind]
    # pred_rel_label = pred_rel_label[pred_rel_sort_ind]
    # pred_rel_pair = pred_rel_pair[pred_rel_sort_ind]
    pred_rel_sort_ind_1 = np.argsort(-pred_rel_score)[:int(len(pred_rel_score) / 3)]
    pred_rel_score = pred_rel_score[pred_rel_sort_ind_1]
    pred_rel_label = pred_rel_label[pred_rel_sort_ind_1]
    pred_rel_pair = pred_rel_pair[pred_rel_sort_ind_1]
    for i in pred_rel_pair:
        gt_new_i0 = gt_old2new[int(i[0])]
        gt_new_i1 = gt_old2new[int(i[1])]
        if gt_new_i0 not in keep_box_idx:
            old2new_idx[int(i[0])] = len(keep_box_idx)
            keep_box_idx.append(gt_new_i0)
        if gt_new_i1 not in keep_box_idx:
            old2new_idx[int(i[1])] = len(keep_box_idx)
            keep_box_idx.append(gt_new_i1)
    keep_box_idx = np.array(keep_box_idx)
    keep_boxes = boxes[keep_box_idx]
    keep_pred_labels = []
    for i in keep_box_idx:
        keep_pred_labels.append(pred_labels[i])
        
    pred_label_num = {}
    pred_count = {}
    for i in keep_pred_labels:
        pred_tmp = idx2label[str(int(i))]
        pred_count[pred_tmp] = 0
        if pred_tmp not in pred_label_num:
            pred_label_num[pred_tmp] = 1
        else:
            pred_label_num[pred_tmp] = pred_label_num[pred_tmp] + 1

    pred_labels_new = []
    for label_i in keep_pred_labels:
        pred_tmp = idx2label[str(int(label_i))]
        if pred_label_num[pred_tmp] == 1:
            pred_labels_new.append(pred_tmp)
        else:
            pred_count[pred_tmp] = pred_count[pred_tmp] + 1
            pred_labels_new.append(str(pred_count[pred_tmp]) + '-' + pred_tmp)


    pred_labels = pred_labels_new

    pred_scores = pred_scores[keep_box_idx]
    pred_scores = pred_scores.tolist()
    pred_rels = [(pred_labels[old2new_idx[int(i[0])]], idx2pred[str(j)], pred_labels[old2new_idx[int(i[1])]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    pred_rels_idx = [(old2new_idx[int(i[0])], idx2pred[str(j)], old2new_idx[int(i[1])]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]

    return img_path, keep_boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, pred_rel_score, pred_rel_label


def get_ground_truth(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    idx2label = vocab_file['idx_to_label']
    idx2pred = vocab_file['idx_to_predicate']
    # boxes
    boxes = groundtruth.bbox
    labels = groundtruth.get_field('labels')
    labels_cls = [idx2label[str(i)] for i in labels.tolist()]
    label_num = {}
    labels_temp = []
    labels_cls_temp = []
    for i in labels_cls:
        if i not in label_num:
            label_num[i] = 1
        else:
            label_num[i] = label_num[i] + 1
        labels_cls_temp.append(i+"_"+str(label_num[i]))
    labels_cls = labels_cls_temp
    #print(labels_cls)
    relation = groundtruth.get_field('relation') # array
    relation_tuple = groundtruth.get_field('relation_tuple') #list: [sub_ind, obj_ind, pred_cls]

    # scores = one_hot_embedding(labels, len(idx2label) + 1)
    # keep_gt_indices = box_nms(boxes, scores, 0.4)
    # keep_gt_indices = keep_gt_indices.numpy()
    # gt_old2new = {}
    #
    # for i, keep_i in enumerate(keep_gt_indices):
    #     gt_old2new[keep_i] = i
    # boxes = boxes[keep_gt_indices]
    # labels = labels[keep_gt_indices]
    rels_temp = []
    relation_tuple = relation_tuple.tolist()
    labels = labels.tolist()
    for i in relation_tuple:
        #if int(i[0]) in keep_gt_indices and int(i[1]) in keep_gt_indices:
            #rels_temp.append([gt_old2new[i[0]],i[2],gt_old2new[i[1]]])
        if [i[0], i[2], i[1]] not in rels_temp:
            rels_temp.append([i[0], i[2], i[1]])
    pred_rels = [(labels_cls[int(i[0])], idx2pred[str(i[1])], labels_cls[int(i[2])]) for i in rels_temp]
    print(pred_rels)
    pred_rels_idx = rels_temp

    return img_path, boxes, labels_cls, pred_rels, pred_rels_idx

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)


def print_list(name, input_list, scores):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i].item()))


def draw_image(img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label,
               print_img=True):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = pred_labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    if print_img:
        display(pic)
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels, None)
        print('*' * 50)
        print_list('gt_rels', gt_rels, None)
        print('*' * 50)
    print_list('pred_labels', pred_labels, pred_rel_score)
    print('*' * 50)
    print_list('pred_rels', pred_rels, pred_rel_score)
    print('*' * 50)

    return None


def draw_scene_graph(img_name, labels, pred_rels=None, pred_rels_idx=None, vis_output_path=None):
    """
    draw a graphviz graph of the scene graph topology
    """
    viz_labels = labels
    viz_rels = pred_rels
    viz_rels_idx = pred_rels_idx
    #s,p,o

    return draw_graph(img_name, viz_labels, viz_rels, viz_rels_idx, vis_output_path)


def draw_graph(img_name, labels, rels, rels_idx, vis_output_path):
    u = Digraph('sg', filename=vis_output_path+img_name+'_sg.gv')
    u.body.append('size="6,6"')
    u.body.append('rankdir="TB"') # LR TB
    # u.body.append('margin=0')
    u.node_attr.update(style='filled')


    name_list = []
    for i, l in enumerate(labels):
        u.node(str(i), label=l, color='#CCCCFF', shape='box', fontsize='36')

    for rel, rel_idx in zip(rels, rels_idx):
        edge_key = '%s_%s' % (rel_idx[0], rel_idx[2])
        u.node(edge_key, label=rel[1], color='#FFCCCC', shape='ellipse',fontsize='36')
        u.edge(str(rel_idx[0]), edge_key)
        u.edge(edge_key, str(rel_idx[2]))
    #u.view()
    u.render(vis_output_path+img_name+'_sg.gv',format='pdf')

def _viz_box(img_name, im, rois, labels, vis_output_path=None):

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    # draw bounding boxes
    for i, bbox in enumerate(rois):
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        label_str = labels[i]
        ax.text(bbox[0], bbox[1] - 4,
                label_str,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=28, color='white')
    ax.axis('off')
    fig.tight_layout()
    print("output: ", vis_output_path)
    plt.savefig(vis_output_path+img_name+'.pdf') #,bbox_inches='tight'
    plt.close()

def draw_boxes_img(img_name, img_path, rois, labels, vis_output_path):
    """
    visualize a scene graph on an image
    """
    viz_rois = rois
    viz_labels = labels
    im = Image.open(img_path)
    return _viz_box(img_name, im, viz_rois, viz_labels, vis_output_path)

def show_selected(idx_list, path_res, vis_output_path):
    for select_idx in idx_list:
        print(f'Image {select_idx}:')
        img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, \
        pred_rel_score, pred_rel_label = get_info_by_idx(
            select_idx, path_res)
        print(img_path)
        # img_path, boxes, pred_labels, pred_rels, pred_rels_idx = get_ground_truth(
        #     select_idx, detected_origin_result)
        #if ('tail' in pred_labels and 'mountain' in pred_labels and 'plane' in pred_labels and 'snow' in pred_labels ): #or \
                #('cow' in pred_labels and 'leg' in pred_labels and 'tail' in pred_labels and 'street' in pred_labels):
        draw_boxes_img(img_name=str(select_idx), img_path=img_path, rois=boxes, labels=pred_labels,
                       vis_output_path=vis_output_path)
        draw_scene_graph(img_name=str(select_idx),labels=pred_labels, pred_rels=pred_rels,
                         pred_rels_idx=pred_rels_idx,
                         vis_output_path=vis_output_path)


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(f'Image {cand_idx}:')
        img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, \
        pred_rel_score, pred_rel_label = get_info_by_idx(
            cand_idx, detected_origin_result)
        draw_boxes_img(img_name=str(cand_idx), img_path=img_path, rois=boxes, labels=pred_labels)
        draw_scene_graph(img_name=str(cand_idx),labels=pred_labels, pred_rels=pred_rel_label, pred_rels_idx=pred_rels_idx)
        # draw_image(img_path=img_path, boxes=boxes, labels=labels, pred_labels=pred_labels, pred_scores=pred_scores,
        #            gt_rels=gt_rels, pred_rels=pred_rels, pred_rel_score=pred_rel_score, pred_rel_label=pred_rel_label,
        #            print_img=True)

def select_from_score():
    eval_results = torch.load(detected_origin_path+'result_dict.pytorch')
    pre_eval_results = torch.load(pre_detected_origin_path + 'result_dict.pytorch')

    recall_list = eval_results['predcls_recall'][20]
    recall_np = np.array(recall_list)
    pre_recall_list = pre_eval_results['predcls_recall'][20]
    pre_recall_np = np.array(pre_recall_list)
    dff_mean_recall = recall_np #- pre_recall_np
    #dff_mean_recall = recall_np
    # pred_out = []
    # for i in pred_idx_list:
    #     mean_recall_np = np.array(eval_results['predcls_mean_recall_collect'][20][i])
    #     pre_mean_recall_np = np.array(pre_eval_results['predcls_mean_recall_collect'][20][i])
    #     #flag_pos = pre_mean_recall_np > 0.5
    #     dff_mean_recall = mean_recall_np - pre_mean_recall_np
    #     pred_out.append(np.argsort(-dff_mean_recall)[:20])
    pred_out = np.argsort(-dff_mean_recall)[:500]
    return pred_out

if __name__ == "__main__":
    #idx_list = select_from_score()
    #idx_list = [24778,24721,10146,18305,3771,6169,21583,16895]
    #idx_list = [24778, 18305, 21583, 15987]
    #idx_list = [11879,19965,22182,2017,3539,3919,9632,9027]
    #idx_list = [22182]
    #idx_list = [9027,16895]
    #idx_list = [9632,3539,3919,16573,19965,10391,25937]
    #idx_list = [4749, 7487, 8055, 8681, 9396, 9632, 11369, 11990, 22888, 23450, 24380, 25703]
    idx_list = [11531]
    global output_path
    # for i, j in zip(pred_list, idx_list):
        # sub_name = i.split(' ')[0]
        # if len(i.split(' ')) > 1:
        #     for k in range(len(i.split(' '))):
        #         if k + 1 >= len(i.split(' ')):
        #             break
        #         sub_name = sub_name + '_' + i.split(' ')[k+1]
        #
        # print(sub_name)
        #output_path = detected_origin_path+'/visualization/' + sub_name + '/'
    #output_path = main_path + '/visualization_good/' #visualization_bad
    mkdir(detected_origin_output_path)
    mkdir(pre_detected_origin_output_path)
    show_selected(idx_list, path_res=detected_origin_result, vis_output_path=detected_origin_output_path)
    show_selected(idx_list, path_res=pre_detected_origin_result, vis_output_path=pre_detected_origin_output_path)
    # show_selected([119, 967, 713, 5224, 19681, 25371])