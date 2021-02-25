import copy 
import torch
import numpy as np 
from models.resnet import resnet18, resnet50, resnet152, resnext101_32x8d, wide_resnet50_2, resnext50_32x4d
from models.resnets import resnet20
from models.densenet import densenet161
from models.shufflenet import shufflenet_v2_x1_0
from models.resnets_2fc import resnet20 as resnet20_2fc
from advertorch.utils import NormalizeByChannelMeanStd
from dataset import *


def setup_model_dataset(args):
    
    if args.dataset == 'cifar10':
        print('training on cifar10 dataset')
        
        if args.arch == 'res18':
            model = resnet18(num_classes=10)
        elif args.arch == 'res50':
            model = resnet50(num_classes=10)
        elif args.arch == 'res20s':
            model = resnet20(number_class=10)
        elif args.arch == 'res20s_2fc':
            model = resnet20_2fc(number_class=10)
        else:
            print('do not support ', args.arch)

        model.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    elif args.dataset == 'cifar10_index':
        print('training on cifar10 subset')

        if args.arch == 'res18':
            model = resnet18(num_classes=10)
        elif args.arch == 'res50':
            model = resnet50(num_classes=10)
        elif args.arch == 'res20s':
            model = resnet20(number_class=10)
        elif args.arch == 'res20s_2fc':
            model = resnet20_2fc(number_class=10)
        else:
            print('do not support ', args.arch)

        model.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders_index(batch_size= args.batch_size, data_dir =args.data, file_name=args.file_name)
    
    elif args.dataset == 'cifar100':
        print('training on cifar100 dataset')

        if args.arch == 'res18':
            print('model = resnet18')
            model = resnet18(num_classes=100)
        elif args.arch == 'res50':
            print('model = resnet50')
            model = resnet50(num_classes=100)
        elif args.arch == 'wide_res50_2':
            print('model = wide_res50_2')
            model = wide_resnet50_2(num_classes=100)
        elif args.arch == 'resnext50_32x4d':
            print('model = resnext50_32x4d')
            model = resnext50_32x4d(num_classes=100)
        elif args.arch == 'res152':
            print('model = resnet152')
            model = resnet152(num_classes=100)
        elif args.arch == 'resnext101':
            print('model = resnext101')
            model = resnext101_32x8d(num_classes=100)
        elif args.arch == 'dense161':
            print('model = dense161')
            model = densenet161(num_classes=100)
        elif args.arch == 'shufflenet':
            print('model = shufflenet_v2_x1_0')
            model = shufflenet_v2_x1_0(num_classes=100)
        elif args.arch == 'res20s':
            print('model = res20s')
            model = resnet20(number_class=100)
        else:
            print('do not support ', args.arch)

        model.normalize = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])

        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    elif args.dataset == 'fmnist':
        
        print('training on fashion-mnist dataset expand to RGB channel')

        if args.arch == 'res18':
            model = resnet18(num_classes=10)
        elif args.arch == 'res50':
            model = resnet50(num_classes=10)
        elif args.arch == 'res20s':
            model = resnet20(number_class=10)
        else:
            print('do not support ', args.arch)

        model.normalize = NormalizeByChannelMeanStd(
            mean=[0.2860, 0.2860, 0.2860], std=[0.3530, 0.3530, 0.3530])
        train_loader, val_loader, test_loader = fashionmnist_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    else:
        print('dataset not support')

    return model, train_loader, val_loader, test_loader

def cvt_state_dict(state_dict, adv_simclr, bn_idx=0):

    state_dict_new = copy.deepcopy(state_dict)

    if adv_simclr:

        for name, item in state_dict.items():

            if 'downsample.conv' in name:
                state_dict_new[name.replace('downsample.conv', 'downsample.0')] = item
                del state_dict_new[name]
                continue

            if 'downsample.bn' in name:
                state_dict_new[name.replace('downsample.bn.bn_list.'+str(bn_idx), 'downsample.1')] = item
                del state_dict_new[name]
                continue
            
            if not 'fc' in name:

                if 'bn_list.'+str(bn_idx) in name:
                    state_dict_new[name.replace('.bn_list.'+str(bn_idx), '')] = item
                    del state_dict_new[name]
                    continue

                if 'bn_list.'+str(1-bn_idx) in name:
                    del state_dict_new[name]
                    continue

            else:
                del state_dict_new[name]

    else:

        for name, item in state_dict.items():

            if 'downsample.conv' in name:
                state_dict_new[name.replace('downsample.conv', 'downsample.0')] = item
                del state_dict_new[name]
                continue

            if 'downsample.bn' in name:
                state_dict_new[name.replace('downsample.bn', 'downsample.1')] = item
                del state_dict_new[name]
                continue

            if 'fc' in name:
                del state_dict_new[name]

    new_dict = {}
    for key in state_dict_new.keys():

        if 'module' in key:
            new_key = key[len('module.'):]
        else:
            new_key = key

        new_dict[new_key] = state_dict_new[key]

    return new_dict

def moco_state_dict(state_dict):
    new_dict = {}

    for key in state_dict.keys():
        if 'module.encoder_q.' in key:
            new_key = key[len('module.encoder_q.'):]
            if not 'fc' in key:
                new_dict[new_key] = state_dict[key]

    return new_dict

def forget_times(record_list):
    
    number = 0
    learned = False

    for i in range(record_list.shape[0]):
        
        if not learned:
            if record_list[i] == 1:
                learned = True 

        else:
            if record_list[i] == 0:
                learned = False
                number+=1 

    return number

def sorted_examples(example_wise_prediction, data_prune, data_rate, state):

    forgetting_events_number = np.zeros(example_wise_prediction.shape[0])
    for j in range(example_wise_prediction.shape[0]):
        tmp_data = example_wise_prediction[j,:]
        if tmp_data[0] < 0:
            forgetting_events_number[j] = -1 
        else:
            forgetting_events_number[j] = forget_times(tmp_data)
    
    if data_prune == 'constent':
        print('* pruning {} data'.format(data_rate))
        rest_number = int(45000*(1-data_rate)**state)
    elif data_prune == 'zero_out':
        print('zero all unforgettable images out')
        rest_number = np.where(forgetting_events_number>0)[0].shape[0]
    else:
        print('error data_prune type')
        assert False

    sequence = np.argsort(forgetting_events_number)[-rest_number:]

    return sequence

def split_class_sequence(sequence, all_labels, num_class):
    
    class_wise_sequence = {}
    for i in range(num_class):
        class_wise_sequence[i] = []
    
    for index in range(sequence.shape[0]):
        class_wise_sequence[all_labels[sequence[index]]].append(sequence[index])
    
    for i in range(num_class):
        class_wise_sequence[i] = np.array(class_wise_sequence[i])
        print('class = {0}, number = {1}'.format(i, class_wise_sequence[i].shape[0]))

    return class_wise_sequence

def blance_dataset_sequence(class_wise_sequence, num_class):
    class_wise_number = np.zeros(num_class, dtype=np.int)
    for i in range(num_class):
        class_wise_number[i] = class_wise_sequence[i].shape[0]
    
    max_length = np.max(class_wise_number)
    print('max class number = {}'.format(max_length))

    balance_sequence = []
    arange_max = np.arange(max_length)
    for i in range(num_class):

        shuffle_index = np.random.permutation(class_wise_number[i])
        shuffle_class_sequence = class_wise_sequence[i][shuffle_index]
        balance_sequence.append(shuffle_class_sequence[arange_max%class_wise_number[i]])

    balance_sequence = np.concatenate(balance_sequence)
    print(balance_sequence.shape)
    return balance_sequence
