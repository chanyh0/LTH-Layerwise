CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 5000 --type random_path --prune-type lt > 0416_resnet50_cifar10_lt_extreme_random_path_5000_GPU7.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 10000 --type random_path --prune-type lt > 0416_resnet50_cifar10_lt_extreme_random_path_10000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 20000 --type random_path --prune-type lt > 0416_resnet50_cifar10_lt_extreme_random_path_20000_GPU1.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 10000 --type ewp --prune-type lt > 0416_resnet50_cifar10_lt_extreme_ewp_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_ewp --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 20000 --type ewp --prune-type lt > 0416_resnet50_cifar10_lt_extreme_ewp_20000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_ewp --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 2000 --prune-type trained --type ewp > 0416_resnet50_cifar10_st_extreme_ewp_2000_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_ewp --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 5000 --prune-type trained --type ewp > 0416_resnet50_cifar10_st_extreme_ewp_5000_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_betweenness --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 200000 --prune-type mt --type betweenness > 0416_resnet50_cifar10_mt_extreme_betweenness_200000_GPU6.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_betweenness --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 1000000 --prune-type mt --type betweenness > 0416_resnet50_cifar10_mt_extreme_betweenness_1000000_GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_st_extreme_random_path --mask_dir resnet50_cifar10_st_0.01/7checkpoint.pth.tar --seed 7 --lr 0.1 --fc --pretrained resnet50_cifar10_st_0.01/7checkpoint.pth.tar --num-paths 2000 --prune-type trained --type random_path > 0416_resnet50_cifar10_st_extreme_random_path_2000_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_ewp --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 20000 --prune-type mt --type ewp > 0416_resnet50_cifar10_mt_extreme_ewp_20000_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_ewp --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 20000 --prune-type pt --type ewp --add-back > 0416_resnet50_cifar100_pt_extreme_ewp_20000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_ewp --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 5000 --prune-type pt --type ewp --add-back > 0416_resnet50_cifar100_pt_extreme_ewp_5000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_ewp --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 10000 --prune-type pt --type ewp --add-back > 0416_resnet50_cifar100_pt_extreme_ewp_10000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 10000 --prune-type pt --type random_path --add-back > 0416_resnet50_cifar100_pt_extreme_random_path_10000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 20000 --prune-type pt --type random_path --add-back > 0416_resnet50_cifar100_pt_extreme_random_path_20000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 5000 --prune-type pt --type random_path --add-back > 0416_resnet50_cifar100_pt_extreme_random_path_5000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 10000 --prune-type pt --type random_path --add-back > 0416_resnet50_cifar100_pt_extreme_random_path_10000_add_back_GPU5.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_pt_extreme_random_path --pretrained pretrained_model/imagenet_weight.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar100_PT/11_checkpoint.pt --num-paths 10000 --prune-type pt --type random_path --add-back > 0416_resnet50_cifar100_pt_extreme_random_path_20000_add_back_GPU6.out &


