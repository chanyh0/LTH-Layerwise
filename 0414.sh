CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 2000 --type random_path --prune-type lt > 0414_resnet50_cifar10_lt_extreme_random_path_2000_GPU7.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 2000 --type ewp --prune-type lt > 0414_resnet50_cifar10_lt_extreme_ewp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 5000 --type ewp --prune-type lt > 0414_resnet50_cifar10_lt_extreme_ewp_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --num-paths 2000 --type random_path --prune-type pt > 0414_resnet50_cifar10_pt_random_path_2000_GPU2.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --num-paths 5000 --type random_path --prune-type pt > 0414_resnet50_cifar10_pt_random_path_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --num-paths 10000 --type random_path --prune-type pt > 0414_resnet50_cifar10_pt_random_path_10000_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_ewp --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --num-paths 2000 --type ewp --prune-type pt > 0414_resnet50_cifar10_pt_ewp_2000_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_ewp --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_PT/18_checkpoint.pt --num-paths 5000 --type ewp --prune-type pt > 0414_resnet50_cifar10_pt_ewp_5000_GPU6.out &

