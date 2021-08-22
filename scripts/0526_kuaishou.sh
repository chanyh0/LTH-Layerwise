CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_taylor1_abs_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 50000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_taylor1_abs_50000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_taylor1_abs_100000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_taylor1_abs_200000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_taylor1_abs_20000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 50000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_taylor1_abs_50000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_taylor1_abs_100000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_taylor1_abs_200000_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar100_lt_extreme_taylor1_abs_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 50000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar100_lt_extreme_taylor1_abs_50000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar100_lt_extreme_taylor1_abs_100000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200000 --type taylor1_abs --prune-type lt  > 0523_res20s_cifar100_lt_extreme_taylor1_abs_200000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_20000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 50000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_50000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_100000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_200000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_20000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_2000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_10000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_20000_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_20000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_2000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_10000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_20000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_20000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_2000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_10000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_20000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_5000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_20000_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_2000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_10000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_20000_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_200_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_100_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_500_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_200_GPU4.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_500_add_back_GPU7.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_200_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_100_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_500_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_200_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_500_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar10 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt > 0523_res18_cifar10_lt_extreme_random_path_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt > 0523_res18_cifar100_lt_extreme_random_path_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt > 0523_res18_cifar100_lt_extreme_random_path_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt > 0523_res18_cifar100_lt_extreme_random_path_500_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_200_GPU4.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_500_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar10 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt > 0523_res18_cifar10_lt_extreme_ewp_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt > 0523_res18_cifar100_lt_extreme_ewp_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt > 0523_res18_cifar100_lt_extreme_ewp_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt > 0523_res18_cifar100_lt_extreme_ewp_500_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_200_GPU4.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_500_add_back_GPU7.out &