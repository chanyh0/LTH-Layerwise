CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.15_w15_OMP_reinit_2_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.15_w15_OMP_reinit_5_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.05_w15_OMP_reinit_5_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.075_w15_OMP_reinit_5_GPU2.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.15_w15_OMP_reinit_3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.15_w15_OMP_reinit_4_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.05_w15_OMP_reinit_4_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.075_w15_OMP_reinit_4_GPU2.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.05_w15_OMP_reinit_3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.075_w15_OMP_reinit_3_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.1_w15_OMP_reinit_3_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.125_w15_OMP_reinit_3_GPU2.out &



CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.05_w15_OMP_reinit_1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.075_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.1_w15_OMP_reinit_1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar10_b128_e160_lr0.125_w15_OMP_reinit_1_GPU2.out &