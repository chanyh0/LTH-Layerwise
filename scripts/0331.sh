CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.05 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.1/3checkpoint.pth.tar > 0331_resnet18_cifar10_lt_0.05_GPU0.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.05 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --prune_type pt > 0331_resnet18_cifar10_pt_0.05_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01 --init pretrained_model/resnet18_lt.pt --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/3checkpoint.pth.tar > 0331_resnet18_cifar100_lt_0.01_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.1/6checkpoint.pth.tar > 0331_resnet18_cifar100_pt_0.01_GPU3.out & 


CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_0.05 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_ST/9_checkpoint.pt --prune_type st > 0331_resnet50_cifar100_st_0.05_GPU4.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.05 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet50_cifar10_st_0.1/7checkpoint.pth.tar --prune_type st > 0331_resnet50_cifar10_st_0.05_GPU5.out &

# titan1

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_0.01 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_MT/0_checkpoint.pt --prune_type mt > 0331_resnet50_cifar100_mt_0.01_GPU0.out &