
from hashlib import new
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
            if index > random_index:
                if name == 'conv1':
                    if conv1:
                        prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
                    else:
                        print('skip conv1 for custom pruning')
                else:
                    prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
                
            else:
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
            print(index)
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
    import random
    for i in range(10000):
        names_to_switch = np.random.choice(names, 2)
        name1 = names_to_switch[0]
        name2 = names_to_switch[1]
        limit = min(random_zeroes[name1], uppers[name2] - random_zeroes[name2])
        to_exchange = random.randint(limit)
        random_zeroes[name1] -= to_exchange
        random_zeroes[name2] += to_exchange

    print(random_zeroes)
    print(sum(random_zeroes.values()))
    index = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index > random_index:
                if name == 'conv1':
                    if conv1:
                        prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
                    else:
                        print('skip conv1 for custom pruning')
                else:
                    prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
                
            else:
                origin_mask = mask_dict[name+'.weight_mask']

                if index <= random_index:
                    number_of_zeros = random_zeroes[name]
                else:
                    number_of_zeros = (origin_mask == 0).sum()
                
                new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = np.random.permutation(new_mask_2).reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=torch.from_numpy(new_mask_2).to(origin_mask.device))
                print((new_mask_2 == 0).sum() / new_mask_2.size)
            print(index)
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
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list+float(m.weight.nelement())
                    zero_sum = zero_sum+float(torch.sum(m.weight == 0))    
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list+float(m.weight.nelement())
                zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

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





