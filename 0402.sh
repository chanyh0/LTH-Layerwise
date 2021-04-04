CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.01 --init pretrained_model/resnet18_pt.pt --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --prune_type pt > 0402_resnet18_cifar10_pt_0.01_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_betweenness.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_betweenness --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 200000 > 0402_resnet18_cifar10_lt_0.1_betweenness_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_betweenness --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 200000 > 0402_resnet18_cifar100_lt_0.01_betweenness_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_betweenness --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 200000 > 0402_resnet18_cifar100_pt_0.01_betweenness_GPU3.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_betweenness --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc > 0402_resnet50_cifar100_mt_0.1_betweenness_GPU0.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_0.01 --init pretrained_model/simclr_weight.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint resnet50_cifar10_st_0.1/7checkpoint.pth.tar --prune_type st > 0402_resnet50_cifar10_st_0.01_GPU5.out &


CUDA_VISIBLE_DEVICES=2 python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_betweenness --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc 

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01 --init LotteryTickets/cifar100_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.01 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_LT/2_checkpoint.pt --prune_type lt > 0401_resnet50_cifar100_lt_0.01_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01_betweenness --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --num-paths 100000 > 0402_resnet50_cifar100_lt_0.01_betweenness_GPU1.out &