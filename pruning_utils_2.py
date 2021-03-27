
from hashlib import new
from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np

def pruning_model(model, px, conv1=True):

    print('start unstructured pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    parameters_to_prune.append((m,'weight'))
                else:
                    print('skip conv1 for L1 unstructure global pruning')
            else:
                parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict, conv1=True, random_index=-1, hold_sparsity = True):

    print('start unstructured pruning with custom mask')
    index = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):

            print("{}: {}".format(index, name))

            if index > random_index:
                print("origin: {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free: {}".format(index))
                number_of_zeros = (mask_dict[name+'.weight_mask'] == 0).sum()
                new_mask = torch.randn(mask_dict[name+'.weight_mask'].shape, device=mask_dict[name+'.weight_mask'].device)
                new_mask_2 = torch.randn(mask_dict[name+'.weight_mask'].shape, device=mask_dict[name+'.weight_mask'].device)
                threshold = np.sort(new_mask.view(-1).cpu().numpy())[number_of_zeros]
                new_mask_2[new_mask <= threshold] = 0
                new_mask_2[new_mask > threshold] = 1
                assert abs((new_mask_2 == 0).sum() - number_of_zeros) < 5 or (not hold_sparsity)
                assert (mask_dict[name+'.weight_mask'] - new_mask_2).abs().mean() > 0 # assert different mask
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2)
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())

            index += 1

def prune_model_custom_random(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index <= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    import random
    for i in range(50000):
        names_to_switch = np.random.choice(names, 2)
        name1 = names_to_switch[0]
        name2 = names_to_switch[1]
        limit = min(random_zeroes[name1], uppers[name2] - random_zeroes[name2])
        to_exchange = random.randint(0, limit)
        random_zeroes[name1] -= to_exchange
        random_zeroes[name2] += to_exchange

    print(random_zeroes)
    print(sum(random_zeroes.values()))
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index > random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                number_of_zeros = random_zeroes[name]
                new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = np.random.permutation(new_mask_2).reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=torch.from_numpy(new_mask_2).to(origin_mask.device))
                print((new_mask_2 == 0).sum() / new_mask_2.size)
            index += 1


def prune_model_custom_random_normal(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index <= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    
    number_of_zeros = sum(random_zeroes.values())
    number_of_elements = sum(uppers.values())

    random_zeroes = list(random_zeroes.values())
    uppers = list(uppers.values())
    indexes = [0]
    for i in range(len(random_zeroes)):
        indexes.append(sum(uppers[:(i+1)]))
    random_values = torch.randn(number_of_elements)
    threshold,_ = torch.topk(random_values, number_of_zeros)
    threshold = threshold[-1]

    new_masks_seq = torch.zeros(number_of_elements)
    new_masks_seq[random_values >= threshold] = 0
    new_masks_seq[random_values < threshold] = 1
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index > random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                #number_of_zeros = random_zeroes[name]
                #new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = new_masks_seq[indexes[index]:indexes[index + 1]].reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2.to(origin_mask.device))
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())
            index += 1


def prune_model_custom_random_normal_reverse(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index >= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    
    number_of_zeros = sum(random_zeroes.values())
    number_of_elements = sum(uppers.values())

    random_zeroes = list(random_zeroes.values())
    uppers = list(uppers.values())
    indexes = [0]
    for i in range(len(random_zeroes)):
        indexes.append(sum(uppers[:(i+1)]))
    random_values = torch.randn(number_of_elements)
    threshold,_ = torch.topk(random_values, number_of_zeros)
    threshold = threshold[-1]

    new_masks_seq = torch.zeros(number_of_elements)
    new_masks_seq[random_values >= threshold] = 0
    new_masks_seq[random_values < threshold] = 1
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index < random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                #number_of_zeros = random_zeroes[name]
                #new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = new_masks_seq[indexes[index - random_index]:indexes[index - random_index + 1]].reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2.to(origin_mask.device))
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())
            index += 1

def remove_prune(model, conv1=True):
    
    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    prune.remove(m,'weight')
                else:
                    print('skip conv1 for remove pruning')
            else:
                prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def reverse_mask(mask_dict):
    new_dict = {}

    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

def extract_main_weight(model_dict, fc=True, conv1=True):
    new_dict = {}

    for key in model_dict.keys():
        if not 'mask' in key:
            if not 'normalize' in key:
                new_dict[key] = model_dict[key]

    if not fc:
        print('delete fc weight')

        delete_keys = []
        for key in new_dict.keys():
            if ('fc' in key) or ('classifier' in key):
                delete_keys.append(key)

        for key in delete_keys:
            del new_dict[key]

    if not conv1:
        print('delete conv1 weight')
        if 'conv1.weight' in new_dict.keys():
            del new_dict['conv1.weight']
        elif 'features.conv0.weight' in new_dict.keys():
            del new_dict['features.conv0.weight']
        elif 'conv1.0.weight' in new_dict.keys():
            del new_dict['conv1.0.weight']

    return new_dict

def check_sparsity(model, conv1=True):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(name)
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list+float(m.weight_mask.nelement())
                    zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))    
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list+float(m.weight_mask.nelement())
                zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def mask_add_back(mask_dict):

    new_mask_dict = {}
    rate_list = []

    for key in mask_dict.keys():

        shape_0 = mask_dict[key].size(0)
        reshape_mask = mask_dict[key].reshape(shape_0, -1)
        zero_number = torch.mean(reshape_mask.eq(0).float(), dim=1)
        rate_list.append(zero_number)

        new_mask = torch.zeros_like(mask_dict[key])
        for indx in range(shape_0):
            if zero_number[indx] != 1:
                new_mask[indx,:] = 1

        new_mask_dict[key] = new_mask

    rate_list = torch.cat(rate_list, dim=0)
    print('all_channels: ', rate_list.shape)
    print('full zero channels: ', torch.sum(rate_list.eq(1).float()))

    return new_mask_dict

def check_zero_channel(mask_dict):

    rate_list = []

    for key in mask_dict.keys():

        shape_0 = mask_dict[key].size(0)
        reshape_mask = mask_dict[key].reshape(shape_0, -1)
        zero_number = torch.mean(reshape_mask.eq(0).float(), dim=1)
        rate_list.append(zero_number)

    rate_list = torch.cat(rate_list, dim=0)
    all_channels_number = rate_list.shape[0]
    zero_channels_number = torch.sum(rate_list.eq(1).float()).item()
    zero_channel_rate = 100*zero_channels_number/all_channels_number 

    print('all_channels: ', all_channels_number)
    print('full zero channels: ', zero_channels_number)
    print('* zero channels rate: {}% '.format(zero_channel_rate))

    return zero_channel_rate




def prune_model_custom_one_random(model, mask_dict, random_index = -1):

    index = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):

            print('pruning layer with custom mask:', name)
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            if index == random_index:
                prune.RandomUnstructured.apply(m, 'weight', amount=(mask_dict[name+'.weight_mask']==0).sum().int().item() / mask_dict[name+'.weight_mask'].numel())
            index += 1

def prune_random_path(model, mask_dict):

    for _ in range(150):
        end_index = None
        for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                print('pruning layer with custom mask:', name)
                try:
                    mask = mask_dict[name+'.weight_mask']
                except:
                    continue
                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                try:
                    if end_index is None:
                        start_index = np.random.randint(0, weight.shape[1] - 1)
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                        else:
                            prob = np.zeros(prob.shape)
                    else:
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                        else:
                            prob = np.zeros(prob.shape)
                except:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                    else:
                        prob = np.zeros(prob.shape)
                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                    else:
                        prob = np.zeros(prob.shape)
                    counter = counter + 1
                    
                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                try:
                    mask = mask_dict[name+'.weight_mask']
                except:
                    continue
                prune.CustomFromMask.apply(m, 'weight', mask=mask)


def prune_random_path_add_back(model, mask_dict):

    n_zeros = 0
    n_param = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
                n_zeros += (mask == 0).float().sum().item()
                n_param += mask.numel()
            except:
                continue

    for _ in range(150):
        end_index = None
        for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                print('pruning layer with custom mask:', name)
                try:
                    mask = mask_dict[name+'.weight_mask']
                except:
                    continue
                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                try:
                    if end_index is None:
                        start_index = np.random.randint(0, weight.shape[1] - 1)
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                        else:
                            prob = np.zeros(prob.shape)
                    else:
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                        else:
                            prob = np.zeros(prob.shape)
                except:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                    else:
                        prob = np.zeros(prob.shape)
                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = (np.abs(prob)>0) / ((np.abs(prob)>0).sum())
                    else:
                        prob = np.zeros(prob.shape)
                    counter = counter + 1
                    
                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    mask_vector = torch.zeros(n_param)
    real_n_zeros = 0
    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
            except:
                continue
            size = np.product(np.array(mask.shape))
            mask_vector[n_cur:n_cur+size] = mask.view(-1)
            n_cur += size
            real_n_zeros += (mask == 0).float().sum().item()
    
    rand_vector = torch.randn(n_param)
    rand_vector[mask_vector == 1] = np.inf
    threshold, _ = torch.kthvalue(rand_vector, int(real_n_zeros - n_zeros))
    mask_vector[rand_vector < threshold] = 1

    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                new_mask = mask_vector[n_cur:n_cur+size].view(mask.shape)
                n_cur += size
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask.to(mask.device))
            except:
                pass
                
                
def prune_random_ewp(model, mask_dict):       
    for _ in range(150):
        end_index = None
        for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                print('pruning layer with custom mask:', name)
                try:
                    mask = mask_dict[name+'.weight_mask']
                except:
                    if 'conv1' in name:
                        continue
                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                try:
                    if end_index is None:
                        start_index = np.random.randint(0, weight.shape[1] - 1)
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = np.abs(prob) / (np.abs(prob).sum())
                        else:
                            prob = np.zeros(prob.shape)
                    else:
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = np.abs(prob) / (np.abs(prob).sum())
                        else:
                            prob = np.zeros(prob.shape)
                except:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = np.abs(prob) / (np.abs(prob).sum())
                    else:
                        prob = np.zeros(prob.shape)

                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = np.abs(prob) / (np.abs(prob).sum())
                    else:
                        prob = np.zeros(prob.shape)

                    counter = counter + 1

                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            except:
                pass


def prune_random_ewp_add_back(model, mask_dict):

    n_zeros = 0
    n_param = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
                n_zeros += (mask == 0).float().sum().item()
                n_param += mask.numel()
            except:
                pass
            
    for _ in range(150):
        end_index = None
        for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                print('pruning layer with custom mask:', name)
                try:
                    mask = mask_dict[name+'.weight_mask']
                except:
                    continue

                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                try:
                    if end_index is None:
                        start_index = np.random.randint(0, weight.shape[1] - 1)
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = np.abs(prob) / (np.abs(prob).sum())
                        else:
                            prob = np.zeros(prob.shape)
                    else:
                        prob = weight[:, start_index]
                        if np.abs(prob).sum() > 0:
                            prob = np.abs(prob) / (np.abs(prob).sum())
                        else:
                            prob = np.zeros(prob.shape)
                except:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = np.abs(prob) / (np.abs(prob).sum())
                    else:
                        prob = np.zeros(prob.shape)

                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = weight[:, start_index]
                    if np.abs(prob).sum() > 0:
                        prob = np.abs(prob) / (np.abs(prob).sum())
                    else:
                        prob = np.zeros(prob.shape)

                    counter = counter + 1

                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    
    mask_vector = torch.zeros(n_param)
    real_n_zeros = 0
    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
            except:
                continue
            size = np.product(np.array(mask.shape))
            mask_vector[n_cur:n_cur+size] = mask.view(-1)
            n_cur += size
            real_n_zeros += (mask == 0).float().sum().item()
    
    rand_vector = torch.randn(n_param)
    rand_vector[mask_vector == 1] = np.inf
    threshold, _ = torch.kthvalue(rand_vector, int(real_n_zeros - n_zeros))
    mask_vector[rand_vector < threshold] = 1

    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                new_mask = mask_vector[n_cur:n_cur+size].view(mask.shape)
                n_cur += size
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask.to(mask.device))
            except:
                pass

def prune_random_betweeness(model, mask_dict):
            
    import networkx

    graph = networkx.Graph()
    name_list = []

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if not 'downsample' in name:
                name_list.append(name)

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) and not 'downsample' in name:
            try:
                mask = mask_dict[name+'.weight_mask']
            except:
                continue
            #prune.CustomFromMask.apply(m, 'weight', mask=mask)
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
    
    edges_betweenness = edge_betweenness_centrality(graph)
    edges_betweenness = sorted((value,key) for (key,value) in edges_betweenness.items())
    for i in range(2000):
        try:
            mask = mask_dict[kernel + '.weight_mask']
        except:
            continue
        edge = edges_betweenness[-i]
        kernel = '.'.join(edge[1][0].split(".")[:-1])
        start_index = int(edge[1][0].split(".")[-1])
        end_index = int(edge[1][1].split(".")[-1])
        
        mask[end_index, start_index] = 0
        mask_dict[kernel + '.weight_mask'] = mask
    
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            try:
                mask = mask_dict[kernel + '.weight_mask']
                prune.CustomFromMask.apply(m, 'weight', mask=mask)
            except:
                continue
            

    


def prune_random_betweeness_add_back(model, mask_dict):
    n_zeros = 0
    n_param = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask = mask_dict[name+'.weight_mask']
            n_zeros += (mask == 0).float().sum().item()
            n_param += mask.numel()

    import networkx

    graph = networkx.Graph()
    name_list = []

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if not 'downsample' in name:
                name_list.append(name)

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d) and not 'downsample' in name:
            mask = mask_dict[name+'.weight_mask']
            #prune.CustomFromMask.apply(m, 'weight', mask=mask)
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

    edges_betweenness = edge_betweenness_centrality(graph)
    edges_betweenness = sorted((value,key) for (key,value) in edges_betweenness.items())
    for i in range(2000):
        edge = edges_betweenness[-i]
        kernel = '.'.join(edge[1][0].split(".")[:-1])
        start_index = int(edge[1][0].split(".")[-1])
        end_index = int(edge[1][1].split(".")[-1])
        mask_dict[kernel + '.weight_mask'][end_index, start_index] = 0
    
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask = mask_dict[name+'.weight_mask']
            prune.CustomFromMask.apply(m, 'weight', mask=mask)

    mask_vector = torch.zeros(n_param)
    real_n_zeros = 0
    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask = mask_dict[name+'.weight_mask']
            size = np.product(np.array(mask.shape))
            mask_vector[n_cur:n_cur+size] = mask.view(-1)
            n_cur += size
            real_n_zeros += (mask == 0).float().sum().item()
    
    rand_vector = torch.randn(n_param)
    rand_vector[mask_vector == 1] = np.inf
    threshold, _ = torch.kthvalue(rand_vector, int(real_n_zeros - n_zeros))
    mask_vector[rand_vector < threshold] = 1

    n_cur = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask = mask_dict[name+'.weight_mask']
            size = np.product(np.array(mask.shape))
            new_mask = mask_vector[n_cur:n_cur+size].view(mask.shape)
            n_cur += size
            prune.CustomFromMask.apply(m, 'weight', mask=new_mask.to(mask.device))