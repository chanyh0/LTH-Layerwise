CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.1 --init LotteryTickets/cifar10_LT/random_init.pt --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_lt_0.1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_pt_0.1.out &

# VITA1
CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1 --init pretrained/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet18_cifar10_lt_0.1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_pt_0.1.out &