CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt > 0523_res20s_cifar10_lt_extreme_hessian_abs_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 50000 --type hessian_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_hessian_abs_50000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100000 --type hessian_abs --prune-type lt  > 0523_res20s_cifar10_lt_extreme_hessian_abs_100000_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_hessian_abs_20000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 50000 --type hessian_abs --prune-type lt --add-back  > 0523_res20s_cifar10_lt_extreme_hessian_abs_50000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100000 --type hessian_abs --prune-type lt --add-back  > 0523_res20s_cifar10_lt_extreme_hessian_abs_100000_add_back_GPU2.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200000 --type hessian_abs --prune-type lt > 0523_res20s_cifar10_lt_extreme_hessian_abs_200000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_hessian_abs_20000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir res50_cifar100_extreme_qrcode_layer1.0.downsample.0.weight_mask --pretrained LotteryTickets/cifar100_LT/random_init.pt --mask_dir ownership/res50_cifar100_qrcode_layer1.0.downsample.0.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity --weight_decay 2e-4 --batch_size 256 --warmup 5 > 0525_res50_cifar100_extreme_qrcode_layer1.0.downsample.0.weight_mask_GPU2.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200000 --type hessian_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_hessian_abs_200000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_hessian_abs_200000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100000 --type hessian_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_hessian_abs_100000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_hessian_abs_100000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 50000 --type hessian_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_hessian_abs_50000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 50000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_hessian_abs_50000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_hessian_abs_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_hessian_abs_20000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_hessian_abs_10000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_betweenness_10000_add_back_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt > 0523_res20s_cifar100_lt_extreme_betweenness_10000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_betweenness_20000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_betweenness_10000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type betweenness --prune-type lt > 0523_res20s_cifar100_lt_extreme_betweenness_20000_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type betweenness --prune-type lt > 0523_res20s_cifar10_lt_extreme_betweenness_5000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_betweenness_5000_add_back_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type betweenness --prune-type lt > 0523_res20s_cifar10_lt_extreme_betweenness_2500_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type taylor1_abs --prune-type lt > 0523_res20s_cifar10_lt_extreme_taylor1_abs_5000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_taylor1_abs_5000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type taylor1_abs --prune-type lt > 0523_res20s_cifar10_lt_extreme_taylor1_abs_2500_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_hessian_abs_5000_add_back_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type hessian_abs --prune-type lt > 0523_res20s_cifar10_lt_extreme_hessian_abs_2500_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type hessian_abs --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_hessian_abs_2500_add_back_GPU1.out &




