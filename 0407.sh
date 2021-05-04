CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_ewp --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 5000 --prune-type trained --type ewp --add-back > 0407_resnet18_cifar10_pt_extreme_ewp_5000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.1 --init pretrained_model/imagenet_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0407_resnet50_cifar10_pt_0.1_seed_1_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.05 --init pretrained_model/imagenet_weight.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0407_resnet50_cifar10_pt_seed_1_0.05_GPU7.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --num-paths 20000 --type ewp --prune-type lt > 0406_resnet50_cifar10_lt_extreme_ewp_20000_GPU0.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_betweenness --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 200000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar10_st_extreme_betweenness_200000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_extreme_betweenness --mask_dir resnet50_cifar100_st_0.01/2checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar100_st_0.01/2checkpoint.pth.tar --num-paths 200000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar100_st_extreme_betweenness_200000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_ewp --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 10000 --prune-type trained --type ewp --add-back > 0407_resnet50_cifar10_st_extreme_ewp_10000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_extreme_ewp --mask_dir resnet50_cifar100_st_0.01/2checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar100_st_0.01/2checkpoint.pth.tar --num-paths 10000 --prune-type trained --type ewp --add-back > 0407_resnet50_cifar100_st_extreme_ewp_10000_add_back_GPU3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01_betweenness --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --num-paths 100000 --type betweenness --prune-type lt --add-back > 0407_resnet50_cifar100_lt_0.01_betweenness_100000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_betweenness --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 200000 --type betweenness --prune-type lt --add-back > 0407_resnet50_cifar10_lt_extreme_betweenness_200000_add_back_GPU0.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_hessian_abs --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 100000 --prune-type trained --type hessian_abs --add-back > 0407_resnet18_cifar100_lt_0.01_hessian_abs_100000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_hessian_abs --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 100000 --prune-type trained --type hessian_abs --add-back > 0407_resnet18_cifar100_pt_0.01_hessian_abs_100000_add_back_GPU3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_hessian_abs --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 100000 --prune-type trained --type hessian_abs --add-back > 0407_resnet18_cifar10_pt_extreme_hessian_abs_100000_add_back_GPU1.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_betweenness --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 1000000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar100_mt_extreme_betweenness_1000000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_betweenness --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 200000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar100_mt_extreme_betweenness_200000_add_back_GPU2.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_random_path --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 2000 --prune-type trained --type random_path > 0408_resnet18_cifar10_lt_extreme_random_path_2000_GPU0.out &