

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_betweenness.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_betweenness --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 50000 > 0402_resnet18_cifar10_lt_0.1_betweenness_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_betweenness.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_betweenness --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 50000 > 0403_resnet18_cifar10_pt_extreme_betweenness_GPU1.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_0.01 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_ST/9_checkpoint.pt --prune_type st > 0403_resnet50_cifar100_st_0.01_GPU4.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_random_path --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 5000 > 0402_resnet18_cifar10_lt_0.1_random_path_GPU0.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_random_path --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 5000 > 0402_resnet18_cifar100_lt_0.01_random_path_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_random_path --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 5000 > 0402_resnet18_cifar100_pt_0.01_random_path_GPU3.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_random_path --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 2000 > 0403_resnet18_cifar10_pt_extreme_random_path_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_0.01 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_LT/7_checkpoint.pt > 0403_resnet50_cifar10_lt_0.01_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01_betweenness --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --num-paths 1000000 > 0402_resnet50_cifar100_lt_0.01_betweenness_GPU1.out &


# ewp
CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_ewp --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 1000 > 0402_resnet18_cifar100_lt_0.01_ewp_GPU2.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_0.01 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar100_ST/9_checkpoint.pt --prune_type st > 0404_resnet50_cifar100_st_0.01_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.01 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet50_cifar10_st_0.1/7checkpoint.pth.tar --prune_type st > 0404_resnet50_cifar10_st_0.01_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_0.01 --init pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_MT/6_checkpoint.pt --prune_type mt > 0401_resnet50_cifar10_mt_0.01_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0403_resnet50_cifar10_pt_0.1_GPU7.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.05 --init pretrained_model/imagenet_weight.pt --seed 13 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0403_resnet50_cifar10_pt_0.1_seed_13_GPU6.out &
