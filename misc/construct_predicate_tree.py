import numpy as np
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
VG_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
conf_mat = np.load('/home/guoyuyu/code/scene_graph_gen/scene_graph_benchmark_pytorch/misc/conf_mat_freq_train.npy')#conf_mat_transformer_train.npy')
pred2ind  = VG_dict['predicate_to_idx']
prd_dict = ['']*(len(pred2ind)+1)
for pred_i in pred2ind.keys():
    prd_dict[pred2ind[pred_i]] = pred_i
prd_dict[0]='NULL'
conf_mat[0,:] = 0
conf_mat[:,0] = 0
def find_father(child, cm, root_list):
    father = np.argmax(cm[child,:])
    #print(father)
    if father not in root_list:
        #print('father!')
        father = find_father(father, cm, root_list)
    return father
        
def find_Connected_Components(conf_mat):
    conf_mat_sum = np.sum(conf_mat, axis=1, keepdims=True)
    cm = conf_mat / (conf_mat_sum.astype(float)+1e-8)
    cm[0,:] = 0
    cm[:,0] = 0
    cm = (cm > 0.2).astype("int") 
    graph = csr_matrix(cm)
    print("graph: ", graph)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    print("n_components: ", n_components)
    print("labels: ", labels)
    comp_list = {}
    for i in range(max(labels)):
        for j in range(len(labels)):
            if labels[j] == i:
                if i not in comp_list:
                    comp_list[i] = []
                comp_list[i].append(prd_dict[j])
    print(comp_list)
    
def mat2forest(conf_mat):
    """
    confusion matrix into forest
    return 
    :param conf_mat: Number_Nodes * Number_Nodes
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    conf_mat_sum = np.sum(conf_mat, axis=1, keepdims=True)
    cm = conf_mat / (conf_mat_sum.astype(float)+1e-8)*100
    # conf_eye = conf_mat.diag()
    # prd_dict = np.array(prd_dict)
    # prd_dist_sort = (0 - conf_eye).argsort()
    # conf_mat_sort = conf_mat[prd_dist_sort,:]
    # conf_mat_sort = conf_mat_sort[:,prd_dist_sort]
    # prd_dist = prd_dist[prd_dist_sort]
    # prd_dict = prd_dict[prd_dist_sort]

    # for i in range(cm.shape[0]):
        # for j in range(cm.shape[1]):
            # if j > i :
                # cm[i,j] = -1
    forest = {}
    for i in range(cm.shape[0]):
        if cm[i,i] == max(cm[i,:]):
            forest[i] = []
    for i in range(cm.shape[0]):
        if i not in forest.keys():
            farther = find_father(i, cm, forest.keys())
            forest[farther].append(i)
    return forest
            
if __name__ == '__main__':
    find_Connected_Components(conf_mat)
    forest = mat2forest(conf_mat)
    forest_dict = {}
    print("Root number: ", len(forest))
    child_count = 0
    for i in forest.keys():
        print('---------father:',prd_dict[i],'---------')
        print('number children: ', len(forest[i]))
        if prd_dict[i] not in forest_dict:
            forest_dict[prd_dict[i]] = []
        for j in forest[i]:
            print(prd_dict[j],end=',')
            child_count = child_count + 1
            forest_dict[prd_dict[i]].append(prd_dict[j])
        print('')
    print(child_count)
    with open('/home/guoyuyu/code/scene_graph_gen/scene_graph_benchmark_pytorch/misc/predicate_forest_label_2layer.json', 'w') as outfile:  
        json.dump(forest_dict, outfile)
    with open('/home/guoyuyu/code/scene_graph_gen/scene_graph_benchmark_pytorch/misc/predicate_forest_ind_2layer.json', 'w') as outfile:  
        json.dump(forest, outfile)