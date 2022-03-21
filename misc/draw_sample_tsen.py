from time import time
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn.manifold import TSNE
from collections import OrderedDict
from matplotlib import colors as mcolors
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import math
from sklearn.decomposition import PCA
# We import seaborn to make nice plots.
import seaborn as sns

from sklearn import preprocessing
import torch

bias_model_res = torch.load("/mnt/hdd3/guoyuyu/datasets/visual_genome/model/sgg_benchmark/checkpoints_best/transformer_predcls_float32_epoch16_batch16/inference_val/VG_stanford_filtered_with_attribute_test/eval_results.pytorch",map_location=torch.device("cpu"))

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
f = '/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/'
dictf = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json','rb'))

type_o = 'edge'  #edge #obj
if type_o == 'obj':
    obj_dict = dictf['label_to_idx']
    #ind_dict = ['tree','table','bike','boy','shirt','banana','clock','laptop']
    ind_dict = ['dog', 'train', 'window',
                'arm', 'banana', 'man',
                'house' ,'flower', 'shirt',
                'sign']
    l_colors = [colors['red'], colors['blue'], colors['orange'],
                colors['green'], colors['black'], colors['deepskyblue'],
                colors['c'], colors['violet'], colors['m'],
                colors['maroon']
                ]
    '''
    ind_dict = obj_dict.keys()
    l_colors = []
    for i in colors.keys():
        l_colors.append(colors[i])
    '''
else:
    obj_dict = dictf['predicate_to_idx']
    no_zero = ['across against', 'attached to', 'growing on', 'standing on', 'to', 'watching']

    ind_dict = ['on','sitting on']
                # 'wearing', \
                # 'looking at','riding','eating',\
                # 'lying on','on back of','with',
                # 'painted on']
    l_colors = [colors['red'], colors['blue']]
                # colors['orange'],
                # colors['green'], colors['black'], colors['deepskyblue'],
                # colors['c'], colors['violet'], colors['m'],
                # colors['maroon']
                # ]
    '''

    ind_dict = obj_dict.keys()
    l_colors = []
    for i in colors.keys():
        l_colors.append(colors[i])
    '''
ind_label = []
for i in ind_dict:
    ind_label.append(obj_dict[i])
ind_label = np.array(ind_label)
def sigmoid(X):

    return 1.0 / (1 + np.exp(-float(X)));



def get_data():
    num_a = 0
    num_b = 0
    label_all = []
    feature_all = {}
    data_s = []
    for i in range(len(bias_model_res['predictions'])):
        gt_rel_tup = bias_model_res['groundtruths'][i].get_field('relation_tuple').long().detach().cpu().numpy()     
        pred_rel_ind = bias_model_res['predictions'][i].get_field('rel_pair_idxs').long().detach().cpu().numpy()
        pred_rel_scores = bias_model_res['predictions'][i].get_field('pred_rel_scores').detach().cpu().numpy()  
        #find_b = np.where(gt_rel_tup[:,2] == ind_label[1])[0]
        for j in range(len(ind_label)):
            find_pred = ind_label[j]
            find_a = np.where(gt_rel_tup[:,2] == find_pred)[0]
            if len(find_a) != 0:
                a_gt_rel_ind = gt_rel_tup[find_a][:,:2]
                diff_ind = np.sum(np.abs(a_gt_rel_ind[:,None,:] - pred_rel_ind[None,:,:]),-1)
                find_pred_ind = np.where(diff_ind == 0)[1]
                pred_rel_find_scores = pred_rel_scores[find_pred_ind][:,1:]
                if len(data_s) == 0:
                    data_s = pred_rel_find_scores
                else:
                    data_s = np.concatenate([data_s,pred_rel_find_scores],0)
                label_t = np.array([j] * len(find_pred_ind))
                if len(label_all) == 0:
                    label_all = label_t
                else:
                    label_all = np.concatenate([label_all,label_t],0)
    
    label = label_all.reshape(-1)
    for j in range(len(ind_label)):
        find_a = np.where(label == j)[0]
        print("Label: ", ind_label[j], "Number: ", len(find_a))
    n_features = data_s.shape[-1]
    n_samples_all = data_s.shape[0]
    return data_s, label, n_samples_all, n_features


def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    '''
    data = preprocessing.scale(data)
    '''
    fig = plt.figure()
    ax = plt.subplot(111)
    #ax = plt.subplot(111, projection='3d')
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=l_colors[label[i]],
                 label=ind_dict[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend(ind_dict)
    return fig


def scatter(x, label):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    colors = []
    label_list = []
    color_float = []
    for i in range(len(ind_dict)):
        color_float.append(mcolors.to_rgba(l_colors[i])[:3])

    for i in range(len(label)):
        colors.append(color_float[label[i]])
        label_list.append(ind_dict[label[i]])

    for i in range(len(ind_dict)):
        find_i = np.where(label==i)[0]
        sc = ax.scatter(x[find_i, 0], x[find_i, 1], lw=0, s=40,
                        c=color_float[i],
                        label=ind_dict[i])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax.legend()
    # We add the labels for each digit.
    txts = []
    '''
    for i in range(len(ind_dict)):
        # Position of each label.
        xtext, ytext = np.median(x[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, ind_dict[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    '''
    return f, ax, sc, txts

def getRandomPointInCircle(num, radius, centerx, centery, color):
    
    samplePoint = []
    for i in range(num):
        theta = random.random() * 2 * np.pi
        r = random.random() * radius
        x = math.cos(theta) * r + centerx
        y = math.sin(theta) * r + centery
        samplePoint.append((int(x), int(y)))
        plt.plot(x, y, '.', color=color, markersize=3)
    return samplePoint

def fgMat_PCA():
    fgmat = np.load("/home/guoyuyu/code/scene_graph_gen/scene_graph_benchmark_pytorch/misc/fg_matrix.npy")
    fgmat = fgmat[1:,1:,1:]
    fgmat = fgmat.reshape([-1, fgmat.shape[-1]])
    fgmat = fgmat.transpose()
    fgmat = fgmat / (fgmat.sum(-1)[:,None] + 1e-8)
    pca = PCA(n_components=2) 
    pca = pca.fit(fgmat)
    X_dr = pca.transform(fgmat)
    pred2idx = dictf['predicate_to_idx']
    idx2ored = dictf['idx_to_predicate']
    idx2color = list(colors.keys())
    plt.figure()
    for i in range(fgmat.shape[0]):
        plt.scatter(X_dr[i, 0] ,X_dr[i, 1]
                    ,alpha=.7
                    ,c=colors[idx2color[i]] ,label=idx2ored[str(i+1)]) 
    plt.legend()
    plt.show()
    
def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=201811, learning_rate=10) #,
    # t0 = time()
    result = tsne.fit_transform(data)
    #fig = plot_embedding(result, label,
    #                     't-SNE embedding of object features')
    scatter(result, label)
    plt.savefig('./misc/fig_'+type_o+'_feat.pdf',format='pdf')

def circular_overlap():
    num = 1000
    radius = 100
    centerx,centery = 20, 20
    samp = getRandomPointInCircle(num, radius, centerx, centery, color="violet")
    num = 1000
    radius = 50
    centerx,centery = 20+100+10, 20
    samp = getRandomPointInCircle(num, radius, centerx, centery, color="deepskyblue")
    plt.axis('off')
    plt.axis('tight')
    plt.savefig('./misc/circle1.pdf',format='pdf')
    

if __name__ == '__main__':
    fgMat_PCA()