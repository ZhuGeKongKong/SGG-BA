#-*-coding:utf-8-*-
import sys
import os
from pprint import pprint
import codecs
import json
from numpy import array
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

path = sys.path[0] + os.sep

def make_mask (shape, dpi) :
    mask_fig = plt.figure(figsize=(6,6),facecolor='w',dpi=dpi)
    mask_ax = mask_fig.add_subplot(111)
    xy_center = (0.5,0.5)
    if (shape == 'circle' or shape == 'c'):
        mask_ax.add_patch(patches.Circle(xy_center, 0.5))
    elif (shape == 'ellipse' or shape == 'e'):
        mask_ax.add_patch(patches.Ellipse(xy_center, 1, 0.75))
    elif (shape == 'rectangle' or shape == 'r'):
        mask_ax.add_patch(patches.Rectangle((0,0.15), 1, 0.7))
    elif (shape == 'square' or shape == 's'):
        mask_ax.add_patch(patches.Rectangle((0,0), 1, 1))
    else :
        shape = int(shape)
        mask_ax.add_patch(patches.RegularPolygon(xy_center, shape, 0.5))
    mask_ax.axis('off')
    mask_fig.savefig('mask.png')
    mask = array(Image.open('mask.png'))
    plt.close()
    return mask

def wc_from_text(str, fn):

    wc = WordCloud( background_color="white",  
        width = 1500, 
        height= 960,  
        margin= 10 ,
        mask = make_mask('e', 200),
        collocations=False,
    ).generate(s)
    plt.imshow(wc)  
    plt.axis("off")  
    plt.show()  # 
    wc.to_file(path + fn)  #

def wc_from_word_count(word_count, fp):

    wc = WordCloud(
        max_words=500,  
        # max_font_size=100,  
        background_color="white",  #
        width = 1500,  # 
        height= 960,  # 
        margin= 10 , # 
        mask = make_mask('e', 200)
    )
    wc.generate_from_frequencies(word_count)  # 
    #plt.imshow(wc)  #
    plt.axis('off')  # 
    #plt.show()  #
    wc.to_file(fp)  #

def generate_dict_from_file(fp):
    with codecs.open(fp, 'r', 'utf-8') as source_file:
        for line in source_file:
            dic = json.loads(line)
            yield dic

def main(data_fp, pic_fp):
    word_count = defaultdict(lambda: 0)
    for dic in generate_dict_from_file(data_fp):
        words = dic['content'].split(' ')
        for word in words:
        	word_count[word] += 1
    with codecs.open(path + 'word_count.json', 'w', 'utf-8') as f:
        json.dump(word_count, f, ensure_ascii=False)
    wc_from_word_count(word_count, pic_fp)

if __name__ == '__main__':
    #wc_from_text(s, 'wc1.jpg')
    vg_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri.json','r'))
    predicates_tree = vg_dict['predicate_count']
    predicates_sort = sorted(predicates_tree.items(), key=lambda x:x[1], reverse=True)
    pred_sort_dict = {}
    pred_count = 0
    pred_num = 10
    for i in predicates_sort:
        if pred_count <= 10:
            pred_sort_dict[i[0]] = i[1] * 3.0/4.0 #8000 #
        else:
            pred_sort_dict[i[0]] = i[1]
        
        pred_count = pred_count + 1
    #pred_sort_dict = predicates_sort
    print(len(pred_sort_dict))
    # pred_sort_dict = json.load(open('clean_predicate_count.json','r'))
    wc_from_word_count(pred_sort_dict, 'predicate_topbottom.jpg')
    # data_fp = path + 'result.json'
    # pic_fp = path + 'word_cloud_uz.jpg'
    # main(data_fp, pic_fp)