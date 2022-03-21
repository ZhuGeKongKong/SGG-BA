import plotly.graph_objects as go
from ipywidgets import widgets
import pandas as pd
import numpy as np
import json

import plotly.express as px
from textwrap import wrap
named_colorscales = px.colors.named_colorscales()

filename = './misc/vis/sooverlap_train_nor0.pdf' 
vg_dict = json.load(open("/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri.json","r"))
ind2pred = vg_dict["idx_to_predicate"]
pred2ind = vg_dict["predicate_to_idx"]
fg_matrix = np.load("./misc/fg_matrix_overlap.npy")
fg_matrix[0, :, :] = 0
fg_matrix[:, 0, :] = 0
fg_matrix[:, :, 0] = 0
fg_matrix[0, 0, 0] = 1
num_rels = fg_matrix.shape[-1]
pred_count = (fg_matrix.reshape((-1, num_rels))).sum(0)

pred_count = np.transpose(pred_count)
pred_count_arg = np.argsort(0.0 - pred_count)
head_pred_all = pred_count_arg[:10]
tail_pred_all = pred_count_arg[10:]
head_pred_lab = ['in' , 'has' ,'of' ,'above']
tail_pred_lab = ['holding', 'parked on', 'along','riding', 'belonging to', 'part of', 'using']
# for i in head_pred:
    # head_pred_lab.append(ind2pred[str(i)])
head_pred = []
for i in head_pred_lab:
    head_pred.append(pred2ind[i])
    
# tail_pred_lab = []
# for i in tail_pred:
    # if i != 0:
        # tail_pred_lab.append(ind2pred[str(i)])

tail_pred = []
for i in tail_pred_lab:
    tail_pred.append(pred2ind[i])

fg_matrix_r = fg_matrix.reshape((-1, fg_matrix.shape[-1]))
fg_matrix_r = fg_matrix_r / (fg_matrix_r.sum(0) + 1e-5)
conf_mat = np.load("./misc/conf_mat_freq_train.npy")
conf_mat_nor1 = conf_mat / (conf_mat.sum(-1) + 1e-8)
bi_graph = np.zeros((num_rels, num_rels))
comm_ic_pair = []
comm_pred = []
comm_pred_ind = []
infor_pred = []
ci_count = []
fg_matrix[fg_matrix<2]=0
for i in head_pred:
    for j in tail_pred:
        if i!=0 and j!= 0:
            bi_graph[i, j] = (
                ((fg_matrix[:, :, i] > 0) * (fg_matrix[:, :, j] > 0)).astype("float")).sum()  # 0.9
            if bi_graph[i, j] > 0:
                comm_pred_i = ind2pred[str(i)]
                infor_pred_j = ind2pred[str(j)]
                comm_pred.append(comm_pred_i)
                comm_pred_ind.append(i)
                infor_pred.append(infor_pred_j)
                ci_count.append(bi_graph[i, j])
        # bi_graph[i,j] = dot(fg_matrix_r[:,i], fg_matrix_r[:,j])/(norm(fg_matrix_r[:,i])*norm(fg_matrix_r[:,j]))
        # bi_graph[i,j] = conf_mat[j,i]
bi_graph[0, :] = 0
bi_graph[:, 0] = 0
bi_graph[0, 0] = 1.0
head_pred_np = np.array(head_pred)
tail_pred_np = np.array(tail_pred)
bi_graph_nor = bi_graph / (bi_graph.sum(0) + 1e-8)
ci_count = (bi_graph_nor[head_pred_np,:])[:, tail_pred_np].reshape(-1)
#bi_graph = bi_graph / (bi_graph.max())
all_colors=['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']
all_colors = ["#f38181","#fce38a","#eaffd0","#95e1d3"]
#all_colors = named_colorscales
head_np = np.array(head_pred)
head_pred_norm = (head_np - head_np.min())/((head_np.max()-head_np.min())*1.0)
color = np.array(comm_pred_ind)
color = (color - head_np.min())/((head_np.max()-head_np.min())*1.0)

colorscale = []
categoryarray = []
head_lap_sort = []
head_pred_norm_sort = np.argsort(head_pred_norm)
for i in range(len(head_pred_norm_sort)):
    categoryarray.append(head_pred_norm[head_pred_norm_sort[i]]) 
    colorscale.append([head_pred_norm[head_pred_norm_sort[i]], all_colors[i]])
    head_lap_sort.append(head_pred_lab[head_pred_norm_sort[i]])

print(color)
print(categoryarray)
print(colorscale)
fig = go.Figure(go.Parcats(
    dimensions=[
        {'label': 'Common Predicates',
         'values': color,
         'categoryarray':categoryarray,
          'ticktext':head_lap_sort,
         },
        {'label': 'Informative Predicates',
         'values': infor_pred}],
    counts=ci_count,
    line={'shape': 'hspline', 'color': color, 'colorscale': colorscale},
    arrangement='freeform'
))
fig.update_layout(
    font_family="Times New Roman",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="black"
)
fig.write_image(filename)
