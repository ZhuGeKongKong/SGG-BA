import h5py
import numpy as np
import json
from collections import defaultdict
import matplotlib as mpl  
import matplotlib.pyplot as plt  
from scipy import stats 
import matplotlib.pylab as pylab
import seaborn as sns
from scipy import stats
import pandas as pd
import csv
vg_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json','r'))
pred_count = vg_dict['predicate_count']
#pred_count = json.load(open('predicate_count.json','r'))
pred_count_wiki = json.load(open('/home/guoyuyu/code/scene_graph_gen/scene_graph_benchmark_pytorch/misc/wikipedia_space/predicate_wikipedia_count.json','r'))
print(pred_count_wiki)
print(len(pred_count_wiki))
pred_count_sort = sorted(pred_count.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
print(len(pred_count_sort))
print(pred_count_sort)
pred_count_s_wiki = []
for i in pred_count_sort:
    pred_count_s_wiki.append([i[0],pred_count_wiki[i[0]]])
name=['predicates','count']
test=pd.DataFrame(columns=name,data=pred_count_s_wiki)
test.to_csv('predicate_dist_wiki.csv',encoding='gbk')
def draw_hist_from_dic(dict, name='None',step=5):
    fig_length = len(dict)
    params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '45',
        'ytick.labelsize': '20',
        'lines.linewidth': '8',
        'legend.fontsize': '25',
        'figure.figsize': str(fig_length)+', 50'  # set figure size
    }
    pylab.rcParams.update(params)
    x = np.arange(len(dict))
    x_labels = []
    y_values = []
    plt.title(name)
    for i in dict:
        y_values.append(i[1])
        x_labels.append(i[0])
    plt.bar(x, y_values)
    plt.xticks(x, x_labels, rotation='vertical', weight=200)
    plt.savefig(name+'.pdf', dpi=200)
    plt.legend(loc='best')
    plt.close('all')
    return 0
def write_cvs():
    f = open('all_predicate_count.json')
    data = json.load(f)
    f.close()

    f = open('all_predicate_count.csv')
    csv_file = csv.writer(f)
    for item in data:
        csv_file.writerow(item)

    f.close()
if __name__ == "__main__":
    print("hahahah")
    #draw_hist_from_dic(dict=pred_count_sort, name='predicate_dist')
    #write_cvs()
