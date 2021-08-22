
CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_structural_lth --pretrained pretrained_model/res20s_cifar10_1.pt --mask_dir admm_0.75.pth.tar_converted --fc > 0822_res20s_cifar10_structural_lth_admm_0.75_original_no_rewind_GPU1.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_structural_lth --pretrained pretrained_model/res20s_cifar10_1.pt --mask_dir admm_0.875.pth.tar_converted --fc > 0822_res20s_cifar10_structural_lth_admm_0.875_original_no_rewind_GPU2.out &
