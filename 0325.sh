CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.01 --init resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.05/8checkpoint.pth.tar > 0325_resnet18_cifar10_lt_0.01_gpu0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.05 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar10_pt_0.2/14checkpoint.pth.tar > 0325_resnet18_cifar10_pt_0.05_gpu1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.05 --init resnet18_cifar100_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.1/13checkpoint.pth.tar > 0325_resnet18_cifar100_lt_0.05_gpu2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.1 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.2_continue/9checkpoint.pth.tar > 0325_resnet18_cifar100_pt_0.1_gpu3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_0.1 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_ST/9_checkpoint.pt > 0324_resnet50_cifar100_st_0.1_gpu4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.1 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_ST/11_checkpoint.pt > 0324_resnet50_cifar10_st_0.1_gpu5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.1 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 5 --resume --checkpoint LotteryTickets/cifar10_MT/6_checkpoint.pt > 0324_resnet50_cifar10_mt_0.1_gpu6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.1 --init pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt > 0324_resnet50_cifar10_pt_0.1_gpu7.out &


