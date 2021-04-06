CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_ewp --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 2000 --prune-type trained --type ewp --add-back > 0405_resnet18_cifar100_lt_0.01_ewp_2000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_ewp --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 2000 --prune-type trained --type ewp --add-back > 0405_resnet18_cifar100_pt_0.01_ewp_2000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_ewp --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 5000 --add-back --prune-type trained --type ewp > 0405_resnet18_cifar10_pt_extreme_ewp_5000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_ewp --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 5000 --prune-type trained --type ewp --add-back > 0405_resnet50_cifar100_mt_extreme_ewp_5000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_mt_extreme_ewp --pretrained resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --mask_dir resnet50_cifar100_mt_0.1/2checkpoint.pth.tar --fc --num-paths 10000 --prune-type trained --type ewp --add-back > 0405_resnet50_cifar100_mt_extreme_ewp_10000_add_back_GPU2.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_betweenness --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 5000 --prune-type trained --type random_path --add-back > 0405_resnet18_cifar100_lt_0.01_random_path_5000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_random_path --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 5000 --prune-type trained --type random_path --add-back > 0405_resnet18_cifar100_pt_0.01_random_path_5000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_random_path --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 5000 --prune-type trained --type random_path > 0405_resnet18_cifar10_pt_extreme_random_path_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01_ewp --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --num-paths 5000 --type ewp --prune-type lt --add-back > 0405_resnet50_cifar100_lt_0.01_ewp_5000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir resnet50_cifar10_lt_extreme_random_path --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --num-paths 10000 --type random_path --prune-type lt --add-back > 0405_resnet50_cifar10_lt_extreme_random_path_10000_add_back_GPU0.out &


