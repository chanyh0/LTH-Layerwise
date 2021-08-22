CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2 --init pretrained_model/resnet18_cifar10_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 20 > 0815_resnet18_cifar10_lt_0.2_gpu0.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_1_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_1_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_2_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_2_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_3_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_3_no_rewind_GPU2.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_4_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_4_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_5_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_5_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_6_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_6_no_rewind_GPU2.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_7_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_7_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_8_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_8_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_9_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_9_no_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained resnet18_cifar10_lt_0.2/0checkpoint.pth.tar --mask_dir new_mask_10_no_rewind.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_structural_lth_10_no_rewind_GPU2.out &