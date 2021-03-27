CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.1 --init LotteryTickets/cifar10_LT/random_init.pt --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_lt_0.1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_pt_0.1.out &

# VITA1
CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2 --init pretrained_model/resnet18_lt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 > resnet18_cifar10_lt_0.2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.2 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 > resnet18_cifar10_pt_0.2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.2 --init pretrained_model/resnet18_lt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 > resnet18_cifar100_lt_0.2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.2 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 > resnet18_cifar100_pt_0.2.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.1 --init LotteryTickets/cifar10_LT/random_init.pt --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_lt_0.1.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_pt_0.1.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.1 --init pretrained_model/simclr_weight.pt --resume --checkpoint LotteryTickets/cifar10_ST/11_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_st_0.1.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.1 --init pretrained_model/moco.pt --resume --checkpoint LotteryTickets/cifar10_MT/5_checkpoint.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar10_mt_0.1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01 --init  LotteryTickets/cifar100_LT/random_init.pt --resume --checkpoint resnet50_cifar100_lt_0.05/9checkpoint.pth.tar --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 > resnet50_cifar100_lt_0.01.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_0.1 --init  pretrained_model/imagenet_weight.pt --resume --checkpoint LotteryTickets/cifar100_PT/11_checkpoint.pt --seed 3 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar100_pt_0.1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_0.1 --init  pretrained_model/moco.pt --resume --checkpoint LotteryTickets/cifar100_MT/5_checkpoint.pt --seed 3 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar100_mt_0.1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_0.05 --init  pretrained_model/moco.pt --resume --checkpoint resnet50_cifar100_mt_0.1/5checkpoint.pth.tar --seed 3 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet50_cifar100_mt_0.05.out &