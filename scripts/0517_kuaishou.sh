CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.2 --init pretrained_model/resnet18_cifar100_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 > 0517_resnet18_cifar100_lt_0.2_gpu6.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.1 --init pretrained_model/resnet18_cifar100_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint resnet18_cifar100_lt_0.2/2checkpoint.pth.tar > 0518_resnet18_cifar100_lt_0.1_gpu6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.1 --init LotteryTickets/cifar100_LT/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_LT/2_checkpoint.pt --prune_type lt > 0527_resnet50_cifar100_lt_0.1_GPU7.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --lr 0.1 --seed 7 --save_model --prune-type lt --type identity > 0517_resnet50_cifar10_lt_extreme_GPU4.out &

CUDA_VISIBLE_DEVICES=4 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir test --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --evaluate --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt 

CUDA_VISIBLE_DEVICES=4 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir test --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --evaluate --checkpoint resnet50_cifar10_lt_extreme/checkpoint.pth.tar --evaluate-p 0.01 --evaluate-random


CUDA_VISIBLE_DEVICES=4 nohup python -u calculate_betweenness.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 7 --lr 0.1 --fc --prune-type lt > 0517_GPU4.out &


# vita4
CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res50 --save_dir resnet50_cifar10_lt_extreme_trigger --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --save_model --seed 7 > 0517_resnet50_cifar10_lt_extreme_trigger_GPU4.out &




