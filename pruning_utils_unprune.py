
from hashlib import new
from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality
import copy
import torch
import networkx
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

def need_to_prune(name, m, conv1):
    return ((name == 'conv1' and conv1) or (name != 'conv1')) \
        and isinstance(m, nn.Conv2d)

def prune(model, mask_dict, prune_type, num_paths=5000, conv1=False, add_back=False):
    new_mask_dict = globals()['prune_' + prune_type](model, mask_dict, num_paths, conv1)
    n_zeros = 0
    n_param = 0
    n_after_zeros = 0
    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):
            mask = mask_dict[name+'.weight_mask']
            n_zeros += (mask == 0).float().sum().item()
            n_param += mask.numel()
            n_after_zeros += new_mask_dict[name+'.weight_mask']
    print("Sparsity before: {}".format(n_zeros / n_param))
    print("Sparsity after: {}".format(n_after_zeros / n_param))
    
    if add_back:
        mask_vector = torch.zeros(n_param)
        n_cur = 0
        for name,m in model.named_modules():
            if need_to_prune(name, m, conv1):
                mask = new_mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                mask_vector[n_cur:n_cur+size] = mask.view(-1)
                n_cur += size
        rand_vector = torch.randn(n_param)
        rand_vector[mask_vector == 1] = np.inf
        threshold, _ = torch.kthvalue(rand_vector, int(n_after_zeros - n_zeros))
        mask_vector[rand_vector < threshold] = 1
        for name,m in model.named_modules():
            if need_to_prune(name, m, conv1):
                mask = mask_dict[name + '.weight_mask']
                prune.CustomFromMask.apply(m, 'weight', mask=mask)
        n_cur = 0
        for name,m in model.named_modules():
            if need_to_prune(name, m, conv1):
                mask = mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                new_mask = mask_vector[n_cur:n_cur+size].view(mask.shape)
                n_cur += size
                m.weight.data = torch.where(new_mask - mask, m.weight.data, torch.randn(mask.shape) / 100)
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask.to(mask.device))
    else:
        for name,m in model.named_modules():
            if need_to_prune(name, m, conv1):
                mask = mask_dict[name + '.weight_mask']
                prune.CustomFromMask.apply(m, 'weight', mask=mask)

def prune_random_path(model, mask_dict, num_paths, conv1=False):
    new_mask_dict = copy.deepcopy(mask_dict)
    for _ in range(num_paths):
        end_index = None
        for name,m in model.named_modules():
            if need_to_prune(name, m, conv1):
                mask = mask_dict[name+'.weight_mask']
                weight = m.weight * mask
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                if end_index is None:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                prob = np.abs(weight[:, start_index]) > 0
                prob = prob / (prob.sum() + 1e-10)

                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index]) > 0
                    prob = prob / (prob.sum() + 1e-10)
                    counter = counter + 1
                    
                    if counter > 200000:
                        prob = np.ones(prob.shape)
                        prob = prob / prob.sum()

                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                new_mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index
    return new_mask_dict

def prune_ewp(model, mask_dict, num_paths, conv1=False):
    new_mask_dict = copy.deepcopy(mask_dict)       
    for _ in range(num_paths):
        end_index = None
        for name,m in model.named_modules():
            print(name)
            if need_to_prune(name, m, conv1):
                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                if end_index is None:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                prob = np.abs(weight[:, start_index])
                prob = prob / (prob.sum() + 1e-10)
                
                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index])
                    prob = prob / (prob.sum() + 1e-10)
                    counter = counter + 1

                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                new_mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    return new_mask_dict
 
def prune_betweeness(model, mask_dict, num_paths, downsample=100, conv1=True):
    new_mask_dict = copy.deepcopy(mask_dict)
    graph = networkx.Graph()
    name_list = []

    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):
            name_list.append(name)

    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):
            mask = mask_dict[name+'.weight_mask']
            weight = mask * m.weight
            weight = torch.sum(weight.abs(), [2, 3])
            for i in range(weight.shape[1]):
                start_name = name + '.{}'.format(i)
                graph.add_node(start_name)
                for j in range(weight.shape[0]):
                    try:
                        end_name = name_list[name_list.index(name) + 1] + '.{}'.format(j)
                        graph.add_node(end_name)
                        
                    except:
                        end_name = 'final.{}'.format(j)
                        graph.add_node(end_name)

                    graph.add_edge(start_name, end_name, weight=weight[j, i])
    
    edges_betweenness = edge_betweenness_centrality(graph, k=int(graph.number_of_nodes() / downsample))
    edges_betweenness = sorted((value,key) for (key,value) in edges_betweenness.items())
    for i in range(num_paths):
        edge = edges_betweenness[-i]
        kernel = '.'.join(edge[1][0].split(".")[:-1])
        start_index = int(edge[1][0].split(".")[-1])
        end_index = int(edge[1][1].split(".")[-1])
        mask = mask_dict[kernel + '.weight_mask']
        mask[end_index, start_index] = 0
        new_mask_dict[kernel + '.weight_mask'] = mask

    return new_mask_dict