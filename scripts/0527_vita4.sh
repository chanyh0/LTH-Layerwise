CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_400_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_400_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_400_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_400_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_100_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_200_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_300_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_400_add_back_GPU3.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_400_GPU3.out &



CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 400 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_400_add_back_GPU7.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 800 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_800_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type ewp --prune-type lt  > 0523_res18_cifar10_lt_extreme_ewp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 800 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_800_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type ewp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_ewp_1000_add_back_GPU3.out &



CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 800 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_800_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type random_path --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_random_path_1000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 800 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_800_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type random_path --prune-type lt  > 0523_res18_cifar10_lt_extreme_random_path_1000_GPU7.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_3000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt  > 0523_res18_cifar10_lt_extreme_omp_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_1000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_2000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_3000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_lt_extreme --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt --add-back > 0523_res18_cifar10_lt_extreme_omp_5000_add_back_GPU7.out &