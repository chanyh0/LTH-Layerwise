CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_lt_random_path.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_random_path --pretrained pretrained_model/imagenet_weight.pt --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_pt_random_path.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_random_path --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar10_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_mt_random_path.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_random_path --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar10_ST/11_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_st_random_path.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_random_path --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_lt_random_path.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_random_path --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar100_ST/9_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_st_random_path.out &
