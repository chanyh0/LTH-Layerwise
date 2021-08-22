num=$1
gpu=$2
CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_${num}_no_rewind.pth.tar --fc > 0822_res18_cifar10_structural_lth_${num}_no_rewind_GPU2.out &
