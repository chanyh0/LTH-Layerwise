CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_8000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_12000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_15000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  --add-back > 0523_res18_cifar10_lt_extreme_omp_8000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_10000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_12000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_15000_add_back_GPU7.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  > 0523_res18_cifar100_lt_extreme_omp_8000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt  > 0523_res18_cifar100_lt_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt  > 0523_res18_cifar100_lt_extreme_omp_12000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt  > 0523_res18_cifar100_lt_extreme_omp_15000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  --add-back > 0523_res18_cifar100_lt_extreme_omp_8000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_10000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_12000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_15000_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 20000 --type omp --prune-type lt  > 0523_res18_cifar100_lt_extreme_omp_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type taylor1_abs --prune-type lt  > 0523_res18_cifar100_lt_extreme_taylor1_abs_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt  > 0523_res18_cifar100_lt_extreme_taylor1_abs_20000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt  > 0523_res18_cifar100_lt_extreme_betweenness_10000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 20000 --type omp --prune-type lt  --add-back > 0523_res18_cifar100_lt_extreme_omp_20000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type taylor1_abs --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_taylor1_abs_10000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 20000 --type taylor1_abs --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_taylor1_abs_20000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_betweenness_10000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt  > 0523_res18_cifar100_lt_extreme_ewp_200_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt  > 0523_res18_cifar100_lt_extreme_ewp_500_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 800 --type ewp --prune-type lt  > 0523_res18_cifar100_lt_extreme_ewp_800_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 1200 --type ewp --prune-type lt  > 0523_res18_cifar100_lt_extreme_ewp_1200_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type omp --prune-type lt  --add-back > 0523_res18_cifar100_lt_extreme_omp_200_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_500_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 800 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_800_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 1200 --type ewp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_ewp_1200_add_back_GPU7.out &

