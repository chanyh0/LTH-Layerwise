import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def pruning_model(model, px, conv1=False):

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                parameters_to_prune.append((m,'weight'))


    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


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

def remove_prune(model, conv1=True):
    print('remove pruning')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                prune.remove(m,'weight')

def extract_mask(model_dict):
    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def extract_main_weight(model_dict):
    new_dict = {}

    for key in model_dict.keys():
        if not 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def prune_model_custom(model, mask_dict, conv1=False):
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                print('pruning layer with custom mask:', name)
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'].to(m.weight.device))

def prune_model_custom_fillback(model, mask_dict, conv1=False):
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                mask = mask_dict[name+'.weight_mask']
                mask = mask.view(mask.shape[0], -1)
                count = torch.sum(mask != 0, 1) # [C]
                #sparsity = torch.sum(mask) / mask.numel()
                num_channel = torch.sum(mask).float() / mask.shape[0]
                int_channel = int(num_channel)
                frac_channel = int_channel - int_channel
                print(mask.shape)
                print(int_channel)
                threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                
                mask[torch.where(count > threshold)[0]] = 1
                mask[torch.where(count < threshold)[0]] = 0
                mask[torch.where(count == threshold)[0],:int(frac_channel * mask.shape[1])] = 1
                mask[torch.where(count == threshold)[0],int(frac_channel * mask.shape[1]):] = 0
                mask = mask.view(*mask_dict[name+'.weight_mask'].shape)
                print('pruning layer with custom mask:', name)
                prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))


def pruning_model_random(model, px):

    print('start unstructured pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

    for name,m in model.named_modules():
        index = 0
        if isinstance(m, nn.Conv2d):            
            origin_mask = m.weight_mask
            print((origin_mask == 0).sum().float() / origin_mask.numel())
            print(index)
            index += 1
            print(name, (origin_mask == 0).sum())