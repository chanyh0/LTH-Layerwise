CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.2 --init pretrained_model/res20s_cifar10_1.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 20 --prune_type lt > 0818_res20s_cifar10_lt_0.2_GPU2.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.2 --init pretrained_model/res20s_cifar100_1.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 20 --prune_type lt > 0818_res20s_cifar100_lt_0.2_GPU3.out &
