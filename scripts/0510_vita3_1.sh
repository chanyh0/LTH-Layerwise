CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir mobilenet_cifar100_b64_e160_lr0.4_w15_random_ticket --pretrained init/mobilenet_cifar100_2.pth.tar --mask_dir mobilenet_cifar100_b64_e160_lr0.4_w15/13checkpoint.pth.tar > 0510_mobilenet_cifar100_0.4_13_random_ticket_IMP_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --epoch 160 --lr 0.4  --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir mobilenet_cifar100_b64_e160_lr0.4_w15_random_ticket --pretrained init/mobilenet_cifar100_2.pth.tar --mask_dir mobilenet_cifar100_b64_e160_lr0.4_w15/14checkpoint.pth.tar > 0510_mobilenet_cifar100_0.4_14_random_ticket_IMP_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_2_GPU3.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_2_GPU3.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_4_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_4_GPU1.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_2_GPU3.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_2_GPU3.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_4_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_4_GPU1.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_rewind_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_rewind_2_GPU3.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_rewind_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_rewind_2_GPU3.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_rewind_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_rewind_4_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_rewind_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_rewind_4_GPU1.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.2_w15_OMP_rewind_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.2_w15_OMP_rewind_2_GPU3.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.4_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.4_w15_OMP_rewind_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.4_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.4_w15_OMP_rewind_2_GPU3.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.2_w15_OMP_rewind_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.2_w15_OMP_rewind_4_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.4_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.4_w15_OMP_rewind_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.4_w15_omp_rewind > 0510_mobilenet_cifar100_b64_e160_lr0.4_w15_OMP_rewind_4_GPU1.out &