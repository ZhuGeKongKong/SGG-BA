import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.font_manager
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
from ipywidgets import widgets


filename = './misc/sooverlap_train_nor0.pdf' 
vg_dict = json.load(open("/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json","r"))
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
head_pred_lab = ['on' , 'has' ,'of' ,'near']
tail_pred_lab = ['standing on', 'attached to', 'along','riding', 'belonging to', 'part of', 'using']
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
B = nx.Graph()
B.add_nodes_from(head_pred_lab, bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from(tail_pred_lab, bipartite=1)

for i in head_pred:
    for j in tail_pred:
        bi_graph[i, j] = (
            ((fg_matrix[:, :, i] > 0) * (fg_matrix[:, :, j] > 0)).astype("float")).sum()  # 0.9
        
        # bi_graph[i,j] = dot(fg_matrix_r[:,i], fg_matrix_r[:,j])/(norm(fg_matrix_r[:,i])*norm(fg_matrix_r[:,j]))
        # bi_graph[i,j] = conf_mat[j,i]
bi_graph[0, :] = 0
bi_graph[:, 0] = 0
bi_graph[0, 0] = 1.0
bi_graph = bi_graph / (bi_graph.max())
#bi_graph = bi_graph / (bi_graph.sum(1)+1e-8)[:,None]
for i in head_pred:
    for j in tail_pred:
        if i != 0 and j != 0:
            if bi_graph[i, j] > 0.1:
                print(ind2pred[str(i)],ind2pred[str(j)],bi_graph[i, j])

print("bi_graph: ", bi_graph.sum(0))
print("bi_graph: ", bi_graph.max())

categorical_dimensions = ['common predicates', 'informative predicates'];

dimensions = [dict(values=cars_df[label], label=label) for label in categorical_dimensions]

# Build colorscale
color = np.zeros(len(cars_df), dtype='uint8')
colorscale = [[0, 'gray'], [1, 'firebrick']]

# Build figure as FigureWidget
fig = go.FigureWidget(
    data=[go.Scatter(x=cars_df.horsepower, y=cars_df['highway-mpg'],
    marker={'color': 'gray'}, mode='markers', selected={'marker': {'color': 'firebrick'}},
    unselected={'marker': {'opacity': 0.3}}), go.Parcats(
        domain={'y': [0, 0.4]}, dimensions=dimensions,
        line={'colorscale': colorscale, 'cmin': 0,
              'cmax': 1, 'color': color, 'shape': 'hspline'})
    ])

fig.update_layout(
        height=800, xaxis={'title': 'Horsepower'},
        yaxis={'title': 'MPG', 'domain': [0.6, 1]},
        dragmode='lasso', hovermode='closest')

# Update color callback
def update_color(trace, points, state):
    # Update scatter selection
    fig.data[0].selectedpoints = points.point_inds

    # Update parcats colors
    new_color = np.zeros(len(cars_df), dtype='uint8')
    new_color[points.point_inds] = 1
    fig.data[1].line.color = new_color

# # Register callback on scatter selection...
# fig.data[0].on_selection(update_color)
# # and parcats click
# fig.data[1].on_click(update_color)

fig





