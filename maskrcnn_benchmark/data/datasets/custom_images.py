import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

BOX_SCALE = 1024  # Scale at which we have the boxes


class GQADataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False,
                 with_clean_classifier=False, cfg=None, custom_eval=False, custom_path=''):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 5000

        assert split in {'train', 'val', 'test'}
        assert self.custom_eval
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.image_ids = load_image_ids(dict_file, split)
        self.ind_to_classes, self.ind_to_attributes, self.ind_to_predicates, \
        self.classes_to_ind, self.attributes_to_ind, self.predicates_to_ind, \
        max_attribute_len = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}
        self.filenames_ori, self.img_info_ori = load_image_filenames(self.image_ids, split, img_dir,
                                                                     image_file)  # length equals to split_mask


        # for i in range(len(img_info1)):
        # if img_info1[i]['width'] != self.img_info[i]['width'] or img_info1[i]['height'] != self.img_info[i]['height']:
        # print(i,img_info1[i]['image_id'])
        self.custom_eval = custom_eval
        self.cfg = cfg
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            self.filenames = [self.filenames_ori[i] for i in np.where(self.split_mask)[0]]
            self.img_info = [self.img_info_ori[i] for i in np.where(self.split_mask)[0]]

    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file,
                                                 dict_file=self.dict_file,
                                                 image_file=self.image_file, must_overlap=False)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        # np.save('./misc/fg_matrix.npy', fg_matrix)
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_image_info = self.img_info_ori  # [:50]#[:50]
        self.custom_files = self.filenames_ori  # [:50] #[:50]
        # if self.cfg.OUTPUT_DIR is not None:
        # for file_name in os.listdir(path):
        #     self.custom_files.append(os.path.join(path, file_name))

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        if self.custom_eval:
            return self.custom_image_info[index]
        else:
            return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index]  # / BOX_SCALE * max(w, h)
        ind_zero = (box[:, 2] - box[:, 0]) == 0 & (box[:, 0] > 0)  # x1 == x2 and x1 > 0
        box[ind_zero, 0] -= 1
        ind_zero = (box[:, 3] - box[:, 1]) == 0 & (box[:, 1] > 0)  # y1 == y2 and y1 > 0
        box[ind_zero, 1] -= 1
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax

        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def load_image_ids(data_path, mode):
    f_mode = mode
    if mode == 'val':
        f_mode = 'train'  # we are using the last 5k training SGs for validation
    elif mode == 'test':
        f_mode = 'val'  # GQA has no public test SGs, so use the val set instead

    img_list_file = os.path.join(data_path, 'questions', '%s_images.json' % f_mode)

    if os.path.isfile(img_list_file):
        print('Loading GQA-%s image ids...' % mode)
        with open(img_list_file, 'r') as f:
            image_ids = json.load(f)
    else:
        # Use only images having question-answer pairs in the balanced split
        print('Loading GQA-%s questions...' % mode)
        with open(os.path.join(data_path, 'questions', '%s_balanced_questions.json' % f_mode), 'rb') as f:
            Q_dict = json.load(f)
        image_ids = set()
        for v in Q_dict.values():
            image_ids.add(v['imageId'])
        with open(img_list_file, 'w') as f:
            json.dump(list(image_ids), f)

        del Q_dict

    image_ids = sorted(list(image_ids))  # sort to make it consistent for different runs
    return image_ids


def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    print("get GQA statistics!!!!!!!!!!!!!!!!!!")
    train_data = GQADataset(split='train', img_dir=img_dir, roidb_file=roidb_file,
                            dict_file=dict_file, image_file=image_file, num_val_im=5000,
                            filter_duplicate_rels=False, with_clean_classifier=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file, output_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    if output_file is not None:
        with open(output_file, 'w') as outfile:
            json.dump(data, outfile)


def load_info(data_path, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(data_path + "sceneGraphs/GQA-SGG-dicts-with-attri.json", 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])
    max_att_len = info['max_len_attribute']
    for i in range(len(ind_to_classes)):
        assert i == info['label_to_idx'][ind_to_classes[i]]
    for i in range(len(ind_to_attributes)):
        assert i == info['attribute_to_idx'][ind_to_attributes[i]]
    for i in range(len(ind_to_predicates)):
        assert i == info['predicate_to_idx'][ind_to_predicates[i]]
    return ind_to_classes, ind_to_attributes, ind_to_predicates, \
           info['label_to_idx'], info['attribute_to_idx'], info['predicate_to_idx'], \
           max_att_len


def load_info_output(data_path, add_bg=True, output_file=None):
    """
    Loads the file containing the GQA label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
             classes_to_ind: map from object classes to indices
             predicates_to_ind: map from predicate classes to indices
    """
    info = {'label_to_idx': {}, 'predicate_to_idx': {}, 'attribute_to_idx': {}}
    label_count = {}
    predicate_count = {}
    attribute_count = {}
    with open(os.path.join(data_path, 'sceneGraphs/train_sceneGraphs.json'), 'rb') as f:
        train_sgs = json.load(f)
    with open(os.path.join(data_path, 'sceneGraphs/val_sceneGraphs.json'), 'rb') as f:
        val_sgs = json.load(f)
    obj_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            if obj['name'] not in label_count:
                label_count[obj['name']] = 0
            label_count[obj['name']] += 1
            obj_classes.add(obj['name'])
    if add_bg:
        ind_to_classes = ['__background__'] + sorted(list(obj_classes))
    else:
        ind_to_classes = sorted(list(obj_classes))
    for obj_lbl, name in enumerate(ind_to_classes):
        info['label_to_idx'][name] = obj_lbl

    rel_classes = set()
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            for rel in obj['relations']:
                if rel['name'] not in predicate_count:
                    predicate_count[rel['name']] = 0
                predicate_count[rel['name']] += 1
                rel_classes.add(rel['name'])
    if add_bg:
        ind_to_predicates = ['__background__'] + sorted(list(rel_classes))
    else:
        ind_to_predicates = sorted(list(rel_classes))
    for rel_lbl, name in enumerate(ind_to_predicates):
        info['predicate_to_idx'][name] = rel_lbl

    att_classes = set()
    max_att_len = 0
    for sg in list(train_sgs.values()) + list(val_sgs.values()):
        for obj in sg['objects'].values():
            if max_att_len <= len(obj['attributes']):
                max_att_len = len(obj['attributes'])
            for att in obj['attributes']:
                if att not in attribute_count:
                    attribute_count[att] = 0
                attribute_count[att] += 1
                att_classes.add(att)
    if add_bg:
        ind_to_attributes = ['__background__'] + sorted(list(att_classes))
    else:
        ind_to_attributes = sorted(list(att_classes))
    for att_lbl, name in enumerate(ind_to_attributes):
        info['attribute_to_idx'][name] = att_lbl
    print("--------object class len: ", len(ind_to_classes))
    print("--------attribute class len: ", len(ind_to_attributes))
    print("--------predicate class len: ", len(ind_to_predicates))
    print("--------max_att_len: ", max_att_len)
    info['idx_to_label'] = ind_to_classes
    info['idx_to_attribute'] = ind_to_attributes
    info['idx_to_predicate'] = ind_to_predicates
    info['label_count'] = label_count
    info['attribute_count'] = attribute_count
    info['predicate_count'] = predicate_count
    info['max_len_attribute'] = max_att_len
    if output_file is not None:
        with open(output_file, 'w') as outfile:
            json.dump(info, outfile)
    return ind_to_classes, ind_to_attributes, ind_to_predicates, \
           info['label_to_idx'], info['attribute_to_idx'], info['predicate_to_idx'], \
           max_att_len


def load_image_filenames(image_ids, mode, image_dir, image_file):
    """
    Loads the image filenames from GQA from the JSON file that contains them.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the GQA images are located
    :return: List of filenames corresponding to the good images
    """

    print("---Start loading image filenames!---")
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    fns = []
    img_info = []
    for im_id in image_ids:
        basename = '{}.jpg'.format(im_id)
        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):  # comment for faster loading
            fns.append(filename)
            img_info.append(im_data[str(im_id)])

    print("---End loading image filenames!---")
    assert len(fns) == len(image_ids), (len(fns), len(image_ids))
    assert len(fns) == (72140 if mode in ['train', 'val'] else 10234), (len(fns), mode)
    assert len(fns) == len(img_info)
    return fns, img_info


def load_graphs(data_path, image_ids, classes_to_ind, attributes_to_ind, predicates_to_ind, num_val_im=-1,
                min_graph_size=-1, max_graph_size=-1, max_attribute_len=-1, mode='train',
                training_triplets=None, random_subset=False,
                filter_empty_rels=True, filter_zeroshots=True,
                exclude_left_right=False, with_clean_classifier=False, ind_to_predicates=None,
                img_info=None):
    """
    Load GT boxes, relations and dataset split
    :param graphs_file_template: template SG filename (replace * with mode)
    :param split_modes_file: JSON containing mapping of image id to its split
    :param mode: (train, val, or test)
    :param training_triplets: a list containing triplets in the training set
    :param random_subset: whether to take a random subset of relations as 0-shot
    :param filter_empty_rels: (will be filtered otherwise.)
    :return: image_index: a np array containing the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    if exclude_left_right:
        print('\n excluding some relationships from GQA!\n')
        filter_rels = []
        for rel in ['to the left of', 'to the right of']:
            filter_rels.append(predicates_to_ind[rel])
        filter_rels = set(filter_rels)
        filter_rels_list = []
        for rel_id_i in filter_rels:
            filter_rels_list.append(ind_to_predicates[rel_id_i])
        print(filter_rels_list)
    if mode in ('train', 'val'):
        with open(os.path.join(data_path, 'sceneGraphs/train_sceneGraphs.json'), 'rb') as f:
            all_sgs_json = json.load(f)
    else:
        with open(os.path.join(data_path, 'sceneGraphs/val_sceneGraphs.json'), 'rb') as f:
            all_sgs_json = json.load(f)

    # Load the image filenames split (i.e. image in train/val/test):
    # train - 0, val - 1, test - 2
    image_index = np.arange(len(image_ids))  # all training/test images
    if num_val_im > 0:
        if mode in ['val']:
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros(len(image_ids)).astype(np.bool)
    split_mask[image_index] = True

    image_idxs = {}
    for i, imid in enumerate(image_ids):
        image_idxs[imid] = i

    # Get everything by SG
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    pred_num = 32
    pred_max_sample = 6000
    pred_ij_dist = {}
    max_w = 0
    max_h = 0
    vg_dict = json.load(open('./datasets/gqa/sceneGraphs/GQA-SGG-dicts-with-attri.json', 'r'))
    predicates_tree = vg_dict['predicate_count']
    predicates_sort = sorted(predicates_tree.items(), key=lambda x: x[1], reverse=True)  # ,
    pred_count = 0
    pred_topk = []
    for pred_i in predicates_sort:
        if pred_count >= pred_num:
            break
        pred_topk.append(str(pred_i[0]))
        pred_count = pred_count + 1
    if with_clean_classifier and mode == 'train':
        root_classes = pred_topk

        for root_i in root_classes:
            pred_ij_dist[root_i] = []
        # bias_model_res = torch.load(
        # "/mnt/data1/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/transformer_predcls_float32_epoch16_batch16/inference_train1/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch",
        # map_location=torch.device("cpu"))
        fg_matrix = np.load("./datasets/gqa/sceneGraphs/fg_matrix.npy")
        fg_matrix[0, :, :] = 0
        fg_matrix[:, 0, :] = 0
        fg_matrix[:, :, 0] = 0
        fg_matrix[0, 0, 0] = 1
        # pred_dist = fg_matrix
        pred_dist = fg_matrix / (fg_matrix.sum(2)[:, :, None] * 1.0 + 1e-8)
    else:
        root_classes = None
    print('split: ', mode)
    print("with_clean_classifier-------------: ", with_clean_classifier)
    print("root_classes-------------: ", root_classes)
    non_match = 0
    rel_c = {}
    for imid in image_ids:

        if not split_mask[image_idxs[imid]]:
            continue

        sg_objects = all_sgs_json[imid]['objects']
        # Sort the keys to ensure object order is always the same
        sorted_oids = sorted(list(sg_objects.keys()))

        # assert filter_empty_rels, 'should filter images with empty rels'

        # Filter out images without objects/bounding boxes
        if len(sorted_oids) == 0:
            split_mask[image_idxs[imid]] = False
            continue

        boxes_i = []
        gt_classes_i = []
        gt_attributes_i = []
        raw_rels = []
        oid_to_idx = {}
        no_objs_with_rels = True
        for oid in sorted_oids:

            obj = sg_objects[oid]

            # Compute object GT bbox
            b = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
            try:
                assert np.all(b[:2] >= 0), (b, obj)  # sanity check
                assert np.all(b[2:] > 0), (b, obj)  # no empty box
            except:
                continue  # skip objects with empty bboxes or negative values

            oid_to_idx[oid] = len(gt_classes_i)
            if len(obj['relations']) > 0:
                no_objs_with_rels = False

            # Compute object GT class
            gt_class = classes_to_ind[obj['name']]
            gt_classes_i.append(gt_class)
            att_i_j = []
            for att_j in obj['attributes']:
                att_i_j.append(attributes_to_ind[att_j])
            att_i_j = att_i_j + [0] * (max_attribute_len - len(att_i_j))
            gt_attributes_i.append(att_i_j)
            # convert to x1, y1, x2, y2
            box = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            # box = np.concatenate((b[:2] - b[2:] / 2, b[:2] + b[2:] / 2))

            boxes_i.append(box)
            # Compute relations from this object to others in the current SG
            for rel in obj['relations']:
                raw_rels.append([oid, rel['object'], rel['name']])  # s, o, r

        # Filter out images without relations - TBD
        if no_objs_with_rels:
            split_mask[image_idxs[imid]] = False
            continue

        if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-10 will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # 11-Inf will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        # Update relations to include SG object ids
        rels = []

        for rel in raw_rels:
            if rel[0] not in oid_to_idx or rel[1] not in oid_to_idx:
                continue  # skip rels for objects with empty bboxes

            R = predicates_to_ind[rel[2]]

            if exclude_left_right:
                if R in filter_rels:
                    continue
            # 50
            rels.append([oid_to_idx[rel[0]],
                         oid_to_idx[rel[1]],
                         R])
            if rel[2] not in rel_c:
                rel_c[rel[2]] = 0
            rel_c[rel[2]] += 1

        rels = np.array(rels)
        n = len(rels)
        if n == 0:
            split_mask[image_idxs[imid]] = False
            continue

        elif training_triplets:
            if random_subset:
                ind_zs = np.random.permutation(n)[:int(np.round(n / 15.))]
            else:
                ind_zs = []
                for rel_ind, tri in enumerate(rels):
                    o1, o2, R = tri
                    tri_str = '{}_{}_{}'.format(gt_classes_i[o1],
                                                R,
                                                gt_classes_i[o2])
                    if tri_str not in training_triplets:
                        ind_zs.append(rel_ind)
                        # print('%s not in the training set' % tri_str, tri)
                ind_zs = np.array(ind_zs)

            if filter_zeroshots:
                if len(ind_zs) > 0:
                    try:
                        rels = rels[ind_zs]
                    except:
                        print(len(rels), ind_zs)
                        raise
                else:
                    rels = np.zeros((0, 3), dtype=np.int32)

            if filter_empty_rels and len(ind_zs) == 0:
                split_mask[image_idxs[imid]] = False
                continue
        if root_classes is not None:
            # pred_rel_scores = bias_model_res['predictions'][i].get_field('pred_rel_scores').detach().cpu().numpy()
            # rel_pair_idxs = bias_model_res['predictions'][i].get_field('rel_pair_idxs').long().detach().cpu().numpy()
            # pred_obj_scores = bias_model_res['predictions'][i].get_field('pred_scores').detach().cpu().numpy()

            for j in range(len(rels)):
                rel_j = rels[j]
                rel_j_pred = ind_to_predicates[rel_j[2]]
                if rel_j_pred in root_classes:
                    # print("rel_dist_j: ", gt_classes_i[rel_j[0]],gt_classes_i[rel_j[1]], rel_j[2])
                    rel_dist_j = pred_dist[gt_classes_i[rel_j[0]], gt_classes_i[rel_j[1]], rel_j[2]]
                    # if len(gt_classes_i) == len(pred_obj_scores):
                    # so_ind = rel_j[:2]
                    # pred_ind = np.where(np.sum(np.abs(so_ind[None,:]-rel_pair_idxs),-1)==0)[0][0]
                    # rel_dist_j = pred_rel_scores[pred_ind][rel_j[2]]
                    # else:
                    # rel_dist_j = 0.0
                    # non_match = non_match + 1
                    # assert rel_dist_j != 0
                    pred_ij_dist[rel_j_pred].append([len(relationships), j, rel_dist_j])
        # Add current SG information to the dataset
        # img_info_i = {}
        # img_info_i['height'] = all_sgs_json[imid]['height']
        # img_info_i['width'] = all_sgs_json[imid]['width']
        # if obj['w'] > img_info[image_idxs[imid]]['width']+2:
        # print('-----', obj['w'], img_info[image_idxs[imid]]['width'])
        # if obj['h'] > img_info[image_idxs[imid]]['height']+2:
        # print('-----',obj['h'], img_info[image_idxs[imid]]['height'])
        boxes_i = np.array(boxes_i)
        gt_classes_i = np.array(gt_classes_i)
        gt_attributes_i = np.array(gt_attributes_i)

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        gt_attributes.append(gt_attributes_i)
    print("rel_c", rel_c)
    # Remove Confusion Samples
    if root_classes is not None and mode == 'train':
        for pred_i in pred_ij_dist.keys():
            pred_ij_dist[pred_i].sort(key=lambda x: x[2], reverse=True)  # Top2000
            # pred_ij_dist[pred_i].sort(key=lambda x: x[2])  # Bottom2000
            pred_ij_dist[pred_i] = np.array(pred_ij_dist[pred_i])
            print(pred_i, len(pred_ij_dist[pred_i]))
            if len(pred_ij_dist[pred_i]) > pred_max_sample:
                pred_ij_dist[pred_i] = pred_ij_dist[pred_i][:pred_max_sample]

                # pred_ij_dist_ind = np.arange(len(pred_ij_dist[pred_i][:,:2]))
                # pred_ij_dist_all = 0.0 - pred_ij_dist[pred_i][:, 2]
                # pred_ij_dist_all = pred_ij_dist_all/(pred_ij_dist_all.sum())
                # pred_ij_dist_rand_ind = np.random.choice(pred_ij_dist_ind, pred_max_sample, replace=False, p=pred_ij_dist_all)
                # pred_ij_dist[pred_i] = pred_ij_dist[pred_i][pred_ij_dist_rand_ind,:2]
            # else:
            # pred_ij_dist[pred_i] = pred_ij_dist[pred_i][:, :2]
        root_classes_count = {}
        leaf_classes_count = {}
        all_classes_count = {}
        image_index = np.where(split_mask)[0]
        rel_ind = -1
        boxes_temp = []
        gt_classes_temp = []
        gt_attributes_temp = []
        relationships_temp = []
        for i in range(len(image_index)):
            if split_mask[image_index[i]]:
                rel_ind = rel_ind + 1
                rel_i = relationships[rel_ind]
                rel_j_temp = []
                for j in range(len(rel_i)):
                    rel_j = rel_i[j]
                    rel_j_pred = ind_to_predicates[rel_j[2]]
                    if rel_j_pred not in all_classes_count:
                        all_classes_count[rel_j_pred] = 1
                    else:
                        all_classes_count[rel_j_pred] = all_classes_count[rel_j_pred] + 1

                    if rel_j_pred in root_classes:
                        ij_ind = np.array([rel_ind, j], dtype='float')
                        if 0 in np.sum(np.abs(ij_ind[None, :] - pred_ij_dist[rel_j_pred][:, :2]), -1):
                            rel_j_temp.append(rel_j)
                            if rel_j_pred not in root_classes_count:
                                root_classes_count[rel_j_pred] = 1
                            else:
                                root_classes_count[rel_j_pred] = root_classes_count[rel_j_pred] + 1
                    else:
                        rel_j_temp.append(rel_j)
                        if rel_j_pred not in leaf_classes_count:
                            leaf_classes_count[rel_j_pred] = 1
                        else:
                            leaf_classes_count[rel_j_pred] = leaf_classes_count[rel_j_pred] + 1
                if len(rel_j_temp) == 0:
                    # print('ignore img index: ', image_index[i])
                    split_mask[image_index[i]] = 0
                    continue
                boxes_temp.append(boxes[rel_ind])
                gt_classes_temp.append(gt_classes[rel_ind])
                gt_attributes_temp.append(gt_attributes[rel_ind])
                relationships_temp.append(np.array(rel_j_temp, dtype=np.int32))
        boxes, gt_classes, gt_attributes, relationships = \
            boxes_temp, gt_classes_temp, gt_attributes_temp, relationships_temp
        print("non match: ", non_match)
        count_list = [0, ]
        for i in root_classes_count:
            count_list.append(root_classes_count[i])
        print('root_classes_count: ', root_classes_count)
        print('mean root class number: ', np.array(count_list).mean())
        print('sum root class number: ', np.array(count_list).sum())

        count_list = [0, ]
        for i in leaf_classes_count:
            count_list.append(leaf_classes_count[i])
        print('leaf_classes_count: ', leaf_classes_count)
        print('mean leaf class number: ', np.array(count_list).mean())
        print('sum leaf class number: ', np.array(count_list).sum())

        count_list = [0, ]
        for i in all_classes_count:
            count_list.append(all_classes_count[i])
        print('all_classes_count: ', all_classes_count)
        print('mean all class number: ', np.array(count_list).mean())
        print('sum all class number: ', np.array(count_list).sum())
    print('number images: ', split_mask.sum())
    print('len boxes: ', len(boxes))
    return split_mask, boxes, gt_classes, gt_attributes, relationships


def load_graphs_nor(data_path, image_ids, classes_to_ind, attributes_to_ind, predicates_to_ind, num_val_im=-1,
                    min_graph_size=-1, max_graph_size=-1, max_attribute_len=-1, mode='train',
                    training_triplets=None, random_subset=False,
                    filter_empty_rels=True, filter_zeroshots=True,
                    exclude_left_right=False, with_clean_classifier=False, ind_to_predicates=None,
                    img_info=None):
    """
    Load GT boxes, relations and dataset split
    :param graphs_file_template: template SG filename (replace * with mode)
    :param split_modes_file: JSON containing mapping of image id to its split
    :param mode: (train, val, or test)
    :param training_triplets: a list containing triplets in the training set
    :param random_subset: whether to take a random subset of relations as 0-shot
    :param filter_empty_rels: (will be filtered otherwise.)
    :return: image_index: a np array containing the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    if exclude_left_right:
        print('\n excluding some relationships from GQA!\n')
        filter_rels = []
        for rel in ['to the left of', 'to the right of']:
            filter_rels.append(predicates_to_ind[rel])
        filter_rels = set(filter_rels)
        filter_rels_list = []
        for rel_id_i in filter_rels:
            filter_rels_list.append(ind_to_predicates[rel_id_i])
        print(filter_rels_list)
    if mode in ('train', 'val'):
        with open(os.path.join(data_path, 'sceneGraphs/train_sceneGraphs.json'), 'rb') as f:
            all_sgs_json = json.load(f)
    else:
        with open(os.path.join(data_path, 'sceneGraphs/val_sceneGraphs.json'), 'rb') as f:
            all_sgs_json = json.load(f)

    # Load the image filenames split (i.e. image in train/val/test):
    # train - 0, val - 1, test - 2
    image_index = np.arange(len(image_ids))  # all training/test images
    if num_val_im > 0:
        if mode in ['val']:
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros(len(image_ids)).astype(np.bool)
    split_mask[image_index] = True

    image_idxs = {}
    for i, imid in enumerate(image_ids):
        image_idxs[imid] = i

    # Get everything by SG
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    pred_num = 32
    pred_max_sample = 10000
    pred_ij_dist = {}
    max_w = 0
    max_h = 0
    vg_dict = json.load(open('./datasets/gqa/sceneGraphs/GQA-SGG-dicts-with-attri.json', 'r'))
    predicates_tree = vg_dict['predicate_count']
    predicates_sort = sorted(predicates_tree.items(), key=lambda x: x[1], reverse=True)  # ,
    pred_count = 0
    pred_topk = []
    for pred_i in predicates_sort:
        if pred_count >= pred_num:
            break
        pred_topk.append(str(pred_i[0]))
        pred_count = pred_count + 1
    if with_clean_classifier and mode == 'train':
        root_classes = pred_topk
    else:
        root_classes = None
    print('split: ', mode)
    print("with_clean_classifier-------------: ", with_clean_classifier)
    print("root_classes-------------: ", root_classes)
    non_match = 0
    rel_c = {}
    root_classes_count = {}
    leaf_classes_count = {}
    all_classes_count = {}
    for imid in image_ids:

        if not split_mask[image_idxs[imid]]:
            continue

        sg_objects = all_sgs_json[imid]['objects']
        # Sort the keys to ensure object order is always the same
        sorted_oids = sorted(list(sg_objects.keys()))

        # assert filter_empty_rels, 'should filter images with empty rels'

        # Filter out images without objects/bounding boxes
        if len(sorted_oids) == 0:
            split_mask[image_idxs[imid]] = False
            continue

        boxes_i = []
        gt_classes_i = []
        gt_attributes_i = []
        raw_rels = []
        oid_to_idx = {}
        no_objs_with_rels = True
        for oid in sorted_oids:

            obj = sg_objects[oid]

            # Compute object GT bbox
            b = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
            try:
                assert np.all(b[:2] >= 0), (b, obj)  # sanity check
                assert np.all(b[2:] > 0), (b, obj)  # no empty box
            except:
                continue  # skip objects with empty bboxes or negative values

            oid_to_idx[oid] = len(gt_classes_i)
            if len(obj['relations']) > 0:
                no_objs_with_rels = False

            # Compute object GT class
            gt_class = classes_to_ind[obj['name']]
            gt_classes_i.append(gt_class)
            att_i_j = []
            for att_j in obj['attributes']:
                att_i_j.append(attributes_to_ind[att_j])
            att_i_j = att_i_j + [0] * (max_attribute_len - len(att_i_j))
            gt_attributes_i.append(att_i_j)
            # convert to x1, y1, x2, y2
            box = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

            # box = np.concatenate((b[:2] - b[2:] / 2, b[:2] + b[2:] / 2))

            boxes_i.append(box)
            # Compute relations from this object to others in the current SG
            for rel in obj['relations']:
                raw_rels.append([oid, rel['object'], rel['name']])  # s, o, r

        # Filter out images without relations - TBD
        if no_objs_with_rels:
            split_mask[image_idxs[imid]] = False
            continue

        if min_graph_size > -1 and len(gt_classes_i) <= min_graph_size:  # 0-10 will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        if max_graph_size > -1 and len(gt_classes_i) > max_graph_size:  # 11-Inf will be excluded
            split_mask[image_idxs[imid]] = False
            continue

        # Update relations to include SG object ids
        rels = []

        for rel in raw_rels:
            if rel[0] not in oid_to_idx or rel[1] not in oid_to_idx:
                continue  # skip rels for objects with empty bboxes

            R = predicates_to_ind[rel[2]]

            if exclude_left_right:
                if R in filter_rels:
                    continue
            # 50
            rels.append([oid_to_idx[rel[0]],
                         oid_to_idx[rel[1]],
                         R])
            if rel[2] not in rel_c:
                rel_c[rel[2]] = 0
            rel_c[rel[2]] += 1

        rels = np.array(rels)
        n = len(rels)
        if n == 0:
            split_mask[image_idxs[imid]] = False
            continue

        elif training_triplets:
            if random_subset:
                ind_zs = np.random.permutation(n)[:int(np.round(n / 15.))]
            else:
                ind_zs = []
                for rel_ind, tri in enumerate(rels):
                    o1, o2, R = tri
                    tri_str = '{}_{}_{}'.format(gt_classes_i[o1],
                                                R,
                                                gt_classes_i[o2])
                    if tri_str not in training_triplets:
                        ind_zs.append(rel_ind)
                        # print('%s not in the training set' % tri_str, tri)
                ind_zs = np.array(ind_zs)

            if filter_zeroshots:
                if len(ind_zs) > 0:
                    try:
                        rels = rels[ind_zs]
                    except:
                        print(len(rels), ind_zs)
                        raise
                else:
                    rels = np.zeros((0, 3), dtype=np.int32)

            if filter_empty_rels and len(ind_zs) == 0:
                split_mask[image_idxs[imid]] = False
                continue
        if root_classes is not None:
            # pred_rel_scores = bias_model_res['predictions'][i].get_field('pred_rel_scores').detach().cpu().numpy()
            # rel_pair_idxs = bias_model_res['predictions'][i].get_field('rel_pair_idxs').long().detach().cpu().numpy()
            # pred_obj_scores = bias_model_res['predictions'][i].get_field('pred_scores').detach().cpu().numpy()
            rel_temp = []
            for j in range(len(rels)):
                rel_j = rels[j]
                rel_j_pred = ind_to_predicates[rel_j[2]]
                if rel_j_pred not in all_classes_count:
                    all_classes_count[rel_j_pred] = 0
                all_classes_count[rel_j_pred] = all_classes_count[rel_j_pred] + 1
                if rel_j_pred not in root_classes or rel_j[2] == 0:
                    rel_j_leaf = rel_j

                    # if rel_i[0] not in boxmap_old2new:
                    # boxmap_old2new[rel_i[0]] = box_num
                    # retain_box.append(rel_i[0])
                    # box_num = box_num + 1
                    # if rel_i[1] not in boxmap_old2new:
                    # boxmap_old2new[rel_i[1]] = box_num
                    # retain_box.append(rel_i[1])
                    # box_num = box_num + 1
                    # rel_i_new[0] = boxmap_old2new[rel_i[0]]
                    # rel_i_new[1] = boxmap_old2new[rel_i[1]]
                    if rel_j_pred not in leaf_classes_count:
                        leaf_classes_count[rel_j_pred] = 0
                    leaf_classes_count[rel_j_pred] = leaf_classes_count[rel_j_pred] + 1
                    rel_temp.append(rel_j_leaf)
                if rel_j_pred in root_classes:
                    rel_j_root = rel_j
                    if rel_j_pred not in root_classes_count:
                        root_classes_count[rel_j_pred] = 0
                    if root_classes_count[rel_j_pred] < pred_max_sample:
                        rel_temp.append(rel_j_root)
                        root_classes_count[rel_j_pred] = root_classes_count[rel_j_pred] + 1
            if len(rel_temp) == 0:
                split_mask[image_idxs[imid]] = False
                continue
            else:
                rels = np.array(rel_temp, dtype=np.int32)
        # Add current SG information to the dataset
        # img_info_i = {}
        # img_info_i['height'] = all_sgs_json[imid]['height']
        # img_info_i['width'] = all_sgs_json[imid]['width']
        # if obj['w'] > img_info[image_idxs[imid]]['width']+2:
        # print('-----', obj['w'], img_info[image_idxs[imid]]['width'])
        # if obj['h'] > img_info[image_idxs[imid]]['height']+2:
        # print('-----',obj['h'], img_info[image_idxs[imid]]['height'])
        boxes_i = np.array(boxes_i)
        gt_classes_i = np.array(gt_classes_i)
        gt_attributes_i = np.array(gt_attributes_i)

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        gt_attributes.append(gt_attributes_i)
    print("rel_c", rel_c)
    # Remove Confusion Samples
    if root_classes is not None:
        print("non match: ", non_match)
        count_list = [0, ]
        for i in root_classes_count:
            count_list.append(root_classes_count[i])
        print('root_classes_count: ', root_classes_count)
        print('mean root class number: ', np.array(count_list).mean())
        print('sum root class number: ', np.array(count_list).sum())

        count_list = [0, ]
        for i in leaf_classes_count:
            count_list.append(leaf_classes_count[i])
        print('leaf_classes_count: ', leaf_classes_count)
        print('mean leaf class number: ', np.array(count_list).mean())
        print('sum leaf class number: ', np.array(count_list).sum())

        count_list = [0, ]
        for i in all_classes_count:
            count_list.append(all_classes_count[i])
        print('all_classes_count: ', all_classes_count)
        print('mean all class number: ', np.array(count_list).mean())
        print('sum all class number: ', np.array(count_list).sum())
    print('number images: ', split_mask.sum())
    print('len boxes: ', len(boxes))
    return split_mask, boxes, gt_classes, gt_attributes, relationships


def convert_img_data_by_id(image_file, output_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    data_by_id = {}
    for i in range(len(data)):
        img = data[i]
        data_by_id[img['image_id']] = img
    with open(output_file, 'w') as outfile:
        json.dump(data_by_id, outfile)
    return


def get_zero_shot_triplet(fg_matrix, output_file):
    zero_shot_triplet = []
    for i in range(1, len(fg_matrix)):
        for j in range(1, len(fg_matrix)):
            for k in range(1, fg_matrix.shape[-1]):
                if fg_matrix[i, j, k] == 0:
                    zero_shot_triplet.append([i, j, k])
    zero_shot_triplet = np.array(zero_shot_triplet)
    np.save(output_file, zero_shot_triplet)
    return


def generate_zero_shot_triplets(fgmat, fgmatval, fgmattest, outputfile):
    fgmat = fgmat[1:, 1:, 1:]
    fgmatval = fgmatval[1:, 1:, 1:]
    fgmattest = fgmattest[1:, 1:, 1:]
    fgmat_zero = (fgmat == 0) * ((fgmatval != 0) + (fgmattest != 0))
    fgmat_zero = np.where(fgmat_zero)
    fgmat_zero_triples = np.concatenate([fgmat_zero[0][:, None], fgmat_zero[1][:, None], fgmat_zero[2][:, None]], -1)
    fgmat_zero_triples = fgmat_zero_triples + 1
    np.save(outputfile + 'zeroshot_triplet.npy', fgmat_zero_triples)


if __name__ == "__main__":
    # ind_to_classes, ind_to_predicates, classes_to_ind, predicates_to_ind = load_info(data_path="/mnt/data1/guoyuyu/datasets/gqa/")
    # print(len(ind_to_classes))
    # print(len(ind_to_predicates))
    # load_info_output(data_path="/mnt/data1/guoyuyu/datasets/gqa/",
    # add_bg=True,
    # output_file="/mnt/data1/guoyuyu/datasets/gqa/sceneGraphs/GQA-SGG-dicts-with-attri.json"
    fg_matrix, bg_matrix = get_VG_statistics(img_dir="/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG_100K/",
                                             roidb_file=None,
                                             dict_file="/mnt/data1/guoyuyu/datasets/gqa/",
                                             image_file="/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/image_data_by_image_id.json",
                                             must_overlap=False)
    output_file = "/mnt/data1/guoyuyu/datasets/gqa/sceneGraphs/"
    # fgmat = np.load(output_file+"fg_matrix.npy")
    # fgmatval = np.load(output_file + "fg_matrix_val.npy")
    # fgmattest = np.load(output_file + "fg_matrix_test.npy")
    # generate_zero_shot_triplets(fgmat, fgmatval, fgmattest, output_file)
    # get_zero_shot_triplet(fg_matrix, output_file+'zeroshot_triplet.npy')
    # np.save(output_file+'fg_matrix.npy', fg_matrix)


