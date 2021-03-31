CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.2/1checkpoint.pth.tar > 0329_resnet18_cifar100_pt_0.1_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.1 --init pretrained_model/resnet18_lt.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/3checkpoint.pth.tar > 0329_resnet18_cifar100_lt_0.1_GPU2.out &


# 2-titan
CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.05 --init LotteryTickets/cifar100_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_LT/2_checkpoint.pt --prune_type lt > 0329_resnet50_cifar100_lt_0.05_GPU1.out &

# titan3
CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_0.05 --init pretrained_model/imagenet_weight.pt --seed 7 --lr 0.05 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_PT/11_checkpoint.pt --prune_type pt > 0329_resnet50_cifar100_pt_0.05_GPU0.out &

# titan1

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_0.05 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_MT/1_checkpoint.pt --prune_type mt > 0329_resnet50_cifar100_mt_0.05_GPU0.out &