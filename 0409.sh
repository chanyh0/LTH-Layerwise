

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_taylor1_abs --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 100000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar10_pt_extreme_taylor1_abs_100000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_ewp --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 1000 --prune-type trained --type ewp --add-back > 0409_resnet18_cifar10_lt_extreme_ewp_1000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_taylor1_abs --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 200000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar100_lt_0.01_taylor1_abs_200000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_taylor1_abs --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 200000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar100_pt_0.01_taylor1_abs_200000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_betweenness --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 1000000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar10_st_extreme_betweenness_1000000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_extreme_betweenness --mask_dir resnet50_cifar100_st_0.01/2checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar100_st_0.01/2checkpoint.pth.tar --num-paths 1000000 --prune-type trained --type betweenness --add-back > 0407_resnet50_cifar100_st_extreme_betweenness_1000000_add_back_GPU5.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_random_path --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 5000 --prune-type trained --type random_path --add-back > 0409_resnet18_cifar10_lt_extreme_random_path_5000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1_intgrads --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 1000000 --prune-type trained --type intgrads --add-back > 0409_resnet18_cifar10_lt_extreme_intgrads_1000000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_taylor1_abs --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 1000000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar100_lt_0.01_taylor1_abs_1000000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_taylor1_abs --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 1000000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar100_pt_0.01_taylor1_abs_1000000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_taylor1_abs --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 1000000 --prune-type trained --type taylor1_abs --add-back > 0409_resnet18_cifar10_pt_extreme_taylor1_abs_1000000_add_back_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_betweenness --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 1000000 --prune-type trained --type betweenness > 0409_resnet50_cifar100_mt_extreme_betweenness_1000000_GPU0.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_betweenness --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 1000000 --prune-type trained --type betweenness --add-back > 0409_resnet50_cifar10_st_extreme_betweenness_1000000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_st_extreme_ewp --mask_dir resnet50_cifar100_st_0.01/2checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar100_st_0.01/2checkpoint.pth.tar --num-paths 10000 --prune-type trained --type ewp > 0409_resnet50_cifar100_st_extreme_ewp_10000_add_back_GPU5.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_pt_0.05 --init pretrained_model/imagenet_weight.pt --seed 13 --lr 0.1 --fc --rate 0.01 --pruning_times 15 --resume --checkpoint LotteryTickets/cifar10_PT/18_checkpoint.pt --prune_type pt > 0407_resnet50_cifar10_pt_seed_13_0.01_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_betweenness --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 200000 --prune-type trained --type betweenness > 0409_resnet50_cifar100_mt_extreme_betweenness_200000_GPU0.out &