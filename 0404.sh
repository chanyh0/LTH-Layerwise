CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_extreme_ewp --mask_dir resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --seed 1 --lr 0.1 --fc --pretrained resnet18_cifar10_pt_0.2/12checkpoint.pth.tar --num-paths 1000 > 0404_resnet18_cifar10_pt_extreme_ewp_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.01_ewp --pretrained resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --mask_dir resnet18_cifar100_lt_0.01/10checkpoint.pth.tar --fc --num-paths 1000 > 0404_resnet18_cifar100_lt_0.01_ewp_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_ewp --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 1000 > 0404_resnet18_cifar100_pt_0.01_ewp_GPU3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.01_random_path --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --num-paths 10000 > 0404_resnet50_cifar100_lt_0.01_random_path_GPU1.out &


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.01_ewp --pretrained resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --mask_dir resnet18_cifar100_pt_0.01/13checkpoint.pth.tar --fc --num-paths 1000 --prune-type trained --type ewp
