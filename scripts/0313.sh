CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_lt_ewp.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_ewp --pretrained pretrained_model/imagenet_weight.pt --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_pt_ewp.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_ewp --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar10_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_mt_ewp.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_ewp --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar10_ST/11_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_st_ewp.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_ewp --pretrained pretrained_model/imagenet_weight.pt --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_pt_ewp.out &
####### TITAN ######
CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_ewp --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_lt_ewp.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_ewp --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar100_ST/9_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_st_ewp.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_ewp --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar100_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_mt_ewp.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_random_path_add_back.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_random_path_add_back --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar100_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_mt_random_path_add_back.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_betweenness --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_lt_betweenness.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_betweenness.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_betweenness --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar100_ST/9_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_st_betweenness.out &




# vita4

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_ewp_add_back --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_lt_ewp_add_back.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_ewp_add_back --pretrained pretrained_model/imagenet_weight.pt --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_pt_ewp_add_back.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_ewp_add_back --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar10_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_mt_ewp_add_back.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_ewp_add_back --pretrained pretrained_model/simclr_weight.pt --mask_dir LotteryTickets/cifar10_ST/11_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar10_st_ewp_add_back.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_ewp_add_back --pretrained pretrained_model/imagenet_weight.pt --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_pt_ewp_add_back.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_ewp_add_back --pretrained pretrained_model/moco.pt --mask_dir LotteryTickets/cifar100_MT/5_checkpoint.pt --seed 1 --conv1 --lr 0.1 --fc > resnet50_cifar100_mt_ewp_add_back.out &