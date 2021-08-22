num=$1
gpu=$2
CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main_eval_fillback.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/${num}checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/${num}checkpoint.pth.tar --fc --prune-type lt > 0820_${num}_GPU${gpu}.out &