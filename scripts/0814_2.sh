CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/10checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/10checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_10_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/9checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/9checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_9_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/8checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/8checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_8_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/7checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/7checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_7_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/6checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/6checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_6_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/5checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/5checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_5_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/4checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/4checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_4_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/3checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/3checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_3_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/2checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/2checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_2_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/1checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/1checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/11checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/11checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_11_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/12checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/12checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_12_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/13checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/13checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_13_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/14checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/14checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_14_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/15checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/15checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_15_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_structural_lth --pretrained LT_cifar10_res18_s1/0checkpoint.pth.tar --mask_dir LT_cifar10_res18_s1/0checkpoint.pth.tar --fc --type identity --prune-type lt > 0814_res18_cifar10_normal_lth_0_GPU7.out &