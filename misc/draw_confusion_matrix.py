import numpy as np
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import matplotlib.font_manager
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from numpy import dot
from numpy.linalg import norm
import torch
# from sklearn.metrics import confusion_matrix
print_type = 'matrix'
model_name = 'transformer_predcls_TopDist15_TopBLMaxDist2k_FixBiGraph_lr1e3_B16'
filename = 'misc/conf_mat_BLRA_SABG.pdf' 
node_names = ["",""]#["on","standing on", "sitting on", "holding","eating","looking at", "near","along"]

figsize = [100,100] 
prd_dist = np.load('misc/rel_dis.npy')
#prd_dist[0] = np.inf
prd_dist = prd_dist[1:]
print('prd_dist shape: ',prd_dist.shape)
#prd_dist = prd_dist[1:]
VG_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
pred2ind  = VG_dict['predicate_to_idx']

def cos_similarity(vec):
    class_num = vec.shape[0]
    cos_sim = np.zeros([class_num,class_num])
    for i in range(class_num):
        for j in range(class_num):
            cos_sim[i][j] = dot(vec[i], vec[j])/(norm(vec[i])*norm(vec[j]))
    return cos_sim

# w2v = np.load('predicates_w2v_mean.npy') 
# print("w2v: ", w2v.sum(-1))
# conf_mat = cos_similarity(w2v)
model_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints/"+model_name+"/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch")
conf_mat = model_res['predicate_confusion_matrix']
#conf_mat = np.load('misc/conf_mat_freq_train.npy')

print(conf_mat.sum(1)[pred2ind['flying in']])
conf_mat[:,0] = 0
conf_mat[0,:] = 0
conf_mat[0,0] = 1.0
print(conf_mat.sum(1)[pred2ind['flying in']])
# conf_mat = conf_mat * (1.0 - np.eye(conf_mat.shape[0]))
# conf_mat = conf_mat + 1.0
conf_mat = conf_mat[1:,1:]
print(conf_mat.sum(1))
pred_list = ['parked on', 'on']

prd_dict = ['']*(len(pred2ind))
for pred_i in pred2ind.keys():
    prd_dict[pred2ind[pred_i]-1] = pred_i
#prd_dict[0]='NULL'
prd_dict = np.array(prd_dict)

conf_mat_nor = conf_mat #conf_mat / (cm_sum.astype(float)[:,None]+1e-8) #* 100
# cm_sum = np.sum(conf_mat, axis=0)
# conf_mat_nor = conf_mat / (cm_sum.astype(float)[None,:] + 1e-8)

cm_sum = np.sum(conf_mat_nor, axis=1)
cm_sum_nor = np.sum(conf_mat_nor, axis=1)
conf_mat_nor = conf_mat_nor / (cm_sum_nor.astype(float)[:,None] + 1e-8) * 100 
# cm_sum_nor = np.sum(conf_mat_nor, axis=0)
# conf_mat_nor = conf_mat_nor / (cm_sum_nor.astype(float)[None, :] + 1e-8) * 100 

prd_dist_sort = (0 - prd_dist).argsort()
conf_mat_nor_sort = conf_mat_nor[prd_dist_sort,:]
conf_mat_nor_sort = conf_mat_nor_sort[:,prd_dist_sort]
print(conf_mat_nor_sort.sum(0))
#print(conf_mat_nor_sort.sum(1))
conf_mat_sort = conf_mat[prd_dist_sort,:]
conf_mat_sort = conf_mat_sort[:,prd_dist_sort]
prd_dist = prd_dist[prd_dist_sort]
prd_dict = prd_dict[prd_dist_sort]
for i in range(len(prd_dict)):
    if prd_dict[i] in pred_list:
        print('i: ',i,'pred: ', prd_dict[i])
    
print("prd_dict: ", prd_dict)
cm_sum_sort = cm_sum[prd_dist_sort]
cm_sum = cm_sum_sort
cm = conf_mat_sort

cm_perc = conf_mat_nor_sort #conf_mat_nor_sort #/ cm_sum.astype(float)*100
annot = np.empty_like(cm).astype(str)
print(cm_perc.sum(1))
nrows, ncols = cm.shape

for i in range(nrows):
    for j in range(ncols):
        p = cm_perc[i, j]
        c = cm[i, j]
        s = cm_sum[i]
        annot[i, j] = '%.1f%%\n%.1f/%d' % (p, c, s)
        
def select_nodes(node_list):
    node_ids = []
    node_names = []
    for i in range(len(prd_dict)):
        if prd_dict[i] in node_list:
            node_ids.append(i)
            node_names.append(prd_dict[i])
    node_ids = np.array(node_ids)
    return node_ids, node_names
    
def select_nodes_num(node_list):
    node_ids = []
    node_names = []
    num = 30
    for i in range(len(prd_dict)):
        if len(node_ids) < 30 and prd_dict[i] not in ['wears','wearing']:
            node_ids.append(i)
            node_names.append(prd_dict[i])
    node_ids = np.array(node_ids)
    return node_ids, node_names
        
#node_ids, node_names = select_nodes_num(node_names)
# cm_perc = cm_perc[node_ids, :][:,node_ids]
# prd_dist = prd_dist[node_ids]
# prd_dict = node_names
# annot = annot[node_ids, :][:,node_ids]
# cm_perc = cm_perc[:30,:30]
# prd_dist = prd_dist[:30]
# prd_dict = prd_dict[:30]
print(cm_perc.shape)
#annot[i, j] = '%.1f' % (p)
# cm_perc_t = np.array([[0.0,1],[1.0,0]])
# cm_perc_t[0:1,0:1] = cm_perc[26:27,0:1]
# prd_dict_a = ['parked on (27)','a']
# prd_dict_b = ['on (1)','b']
cm = pd.DataFrame(cm_perc, index=prd_dict, columns=prd_dict)
print(cm)
#annot = np.empty_like(cm_perc).astype(str)
#annot[0, 0]=r'$c_{34,5}$'
if print_type == 'matrix':
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=8)
    res = sns.heatmap(cm, annot=None, fmt='', ax=None, annot_kws={}, cbar=False)  
    res.set_yticklabels(res.get_ymajorticklabels(),va='center', fontsize = 64, rotation=0, family = 'Times New Roman')
    res.set_xticklabels(res.get_xmajorticklabels(), ha='center',fontsize = 64, rotation=90, family = 'Times New Roman') 
    # plt.xlabel('Output', fontsize = 64) # x-axis label with fontsize 15
    # plt.ylabel('Annotation', fontsize = 64) 
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    
if print_type == 'graph':
    #G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    # cm = cm * (1.0* (cm>0))
    # cm = cm[:10,:10]
    G = nx.from_pandas_adjacency(cm, create_using=nx.DiGraph)
    pos = nx.layout.circular_layout(G)
    prd_dist_nor = prd_dist / prd_dist.sum()
    node_sizes = [i*100 for i in prd_dist_nor]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='r')
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
