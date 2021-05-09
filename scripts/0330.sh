CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.2/14checkpoint.pth.tar > 0330_resnet18_cifar10_lt_0.1_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --prune_type pt > 0330_resnet18_cifar10_pt_0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.05 --init pretrained_model/resnet18_pt.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.1/6checkpoint.pth.tar > 0330_resnet18_cifar100_pt_0.05_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.05 --init pretrained_model/resnet18_lt.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/3checkpoint.pth.tar > 0330_resnet18_cifar100_lt_0.05_GPU2.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.05 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_MT/6_checkpoint.pt --prune_type mt > 0330_resnet50_cifar10_mt_0.05_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.05 --init pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0330_resnet50_cifar10_pt_0.05_GPU7.out &


# 2-titan
CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.1 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt > 0330_resnet50_cifar10_lt_0.1_GPU0.out
