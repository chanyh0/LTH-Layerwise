CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_0.1 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_MT/1_checkpoint.pt --prune_type mt > 0328_resnet50_cifar100_mt_0.1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.1 --init LotteryTickets/cifar100_LT/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_LT/2_checkpoint.pt --prune_type lt > 0328_resnet50_cifar100_lt_0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_0.1 --init pretrained_model/imagenet_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_PT/11_checkpoint.pt --prune_type pt > 0328_resnet50_cifar100_pt_0.1_GPU0.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0328_resnet50_cifar10_pt_0.1_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.1 --init LotteryTickets/cifar10_LT/random_init.pt --seed 13 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt > 0328_resnet50_cifar10_lt_0.1_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1 --init resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.2/7checkpoint.pth.tar > 0327_resnet18_cifar10_lt_0.1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > 0327_resnet18_cifar10_pt_0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.1 --init resnet18_cifar100_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/6checkpoint.pth.tar > 0327_resnet18_cifar100_lt_0.1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.2/14checkpoint.pth.tar > 0327_resnet18_cifar100_pt_0.1_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_0.1 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_ST/9_checkpoint.pt --prune_type st > 0328_resnet50_cifar100_st_0.1_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.1 --init pretrained_model/simclr_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_ST/11_checkpoint.pt --prune_type st > 0328_resnet50_cifar10_st_0.1_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.1 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_MT/6_checkpoint.pt --prune_type mt > 0328_resnet50_cifar10_mt_0.1_GPU6.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt > 0328_resnet18_cifar10_lt_0.2_GPU0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.2 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type pt > 0328_resnet18_cifar10_pt_0.2_GPU1.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.2 --init pretrained_model/resnet18_lt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt > 0328_resnet18_cifar100_lt_0.2_GPU2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.2 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type pt > 0328_resnet18_cifar100_pt_0.2_GPU3.out &