

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_ewp --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 5000 --prune-type mt --type ewp > 0412_resnet50_cifar10_mt_extreme_ewp_5000_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_ewp --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 20000 --prune-type mt --type ewp > 0412_resnet50_cifar10_mt_extreme_ewp_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_ewp --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 10000 --prune-type mt --type ewp --add-back > 0412_resnet50_cifar10_mt_extreme_ewp_10000_add_back_GPU2.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_random_path --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 2000 --prune-type mt --type random_path --add-back > 0412_resnet50_cifar10_mt_extreme_random_path_2000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_random_path --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 5000 --prune-type mt --type random_path --add-back > 0412_resnet50_cifar10_mt_extreme_random_path_5000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_random_path --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 10000 --prune-type mt --type random_path --add-back > 0412_resnet50_cifar10_mt_extreme_random_path_10000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_mt_extreme_betweenness --pretrained pretrained_model/moco.pt --seed 7 --lr 0.1 --fc --mask_dir LotteryTickets/cifar10_MT/6_checkpoint.pt --num-paths 100000 --prune-type mt --type betweenness --add-back > 0412_resnet50_cifar10_mt_extreme_betweenness_100000_add_back_GPU6.out &

