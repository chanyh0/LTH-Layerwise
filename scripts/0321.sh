CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.05 --init resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.1/12checkpoint.pth.tar > 0321_resnet18_cifar10_lt_0.05_gpu0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar10_pt_0.2/14checkpoint.pth.tar > 0321_resnet18_cifar10_pt_0.1_gpu1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.1 --init resnet18_cifar100_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/6checkpoint.pth.tar > 0321_resnet18_cifar100_lt_0.1_gpu2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.2_continue --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.2/14checkpoint.pth.tar > 0320_resnet18_cifar100_pt_0.2_gpu3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1 --init resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.2/7checkpoint.pth.tar > resnet18_cifar10_lt_0.1.out 