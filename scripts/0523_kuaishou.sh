CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_2000_GPU4.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_5000_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_10000_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type betweenness --prune-type lt > 0523_res20s_cifar10_lt_extreme_betweenness_20000_GPU2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type betweenness --prune-type lt > 0523_res20s_cifar10_lt_extreme_betweenness_10000_GPU4.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_2000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_5000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_10000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_2000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_0.1 --batch_size 256 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.1 --pruning_times 17 --prune_type rewind_lt --rewind_epoch 3 --warmup 5 --weight_decay 2e-4 --resume --checkpoint res50_cifar10_lt_0.2/7checkpoint.pth.tar > 0523_res50_cifar10_rewind_lt_0.1_GPU1.out &


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar100_lt_0.2_continue/18model_SA_best.pth.tar


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_2000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_5000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_10000_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_2000_add_back_GPU4.out &


CUDA_VISIBLE_DEVICES=5 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar10_lt_0.01/27model_SA_best.pth.tar

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt > 0523_res20s_cifar10_lt_extreme_random_path_20000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt > 0523_res20s_cifar10_lt_extreme_ewp_20000_GPU3.out &





 