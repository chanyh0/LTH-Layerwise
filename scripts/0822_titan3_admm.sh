
CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir admm_0.75.pth.tar --fc > 0822_res18_cifar10_structural_lth_admm_0.75_no_rewind_GPU1.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir admm_0.875.pth.tar --fc > 0822_res18_cifar10_structural_lth_admm_0.875_no_rewind_GPU2.out &
