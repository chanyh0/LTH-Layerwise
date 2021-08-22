
CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_10.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_1.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_1_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_10.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_10_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_9.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_9_GPU2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_6.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_6_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_5.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_5_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_4.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_4_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_3.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_3_GPU2.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_2.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_2_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_11.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_11_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir new_mask_12.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_12_GPU2.out &